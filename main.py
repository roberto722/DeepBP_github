# -*- coding: utf-8 -*-

import os
import json
import math
import random
import shutil
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import HDF5Dataset, minmax_scale

# ================================
# Utilities & Repro
# ================================

def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make convolutions deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# Geometry configuration (Linear Probe)
# ================================

@dataclass
class LinearProbeGeom:
    """Geometry parameters for a linear transducer array."""
    n_det: int                 # number of detector elements
    pitch_m: float             # center-to-center spacing between elements [m]
    t0_s: float                # time-of-flight start [s] (e.g., recording start)
    dt_s: float                # sampling interval [s]
    n_t: int                   # number of temporal samples
    c_m_s: float               # speed of sound [m/s]
    x0_m: float                # image left border x (meters)
    y0_m: float                # image top border y (meters)
    dx_m: float                # pixel size x [m]
    dy_m: float                # pixel size y [m]
    nx: int                    # image width in pixels
    ny: int                    # image height in pixels
    array_x0_m: float = 0.0    # x-position of the first element center [m]
    array_y_m: float = 0.0     # y-position (depth) of the array line [m], typically y=0

    @property
    def det_x(self) -> np.ndarray:
        """Detector x positions along the array (1D)."""
        # Centered around array_x0_m
        xs = self.array_x0_m + np.arange(self.n_det) * self.pitch_m
        return xs

    @property
    def det_y(self) -> np.ndarray:
        """Detector y positions (all on same y for linear array)."""
        return np.full((self.n_det,), self.array_y_m, dtype=np.float32)

# ================================
# Delay-and-Sum LUT builder
# ================================

def build_delay_and_sum_lut(geom: LinearProbeGeom, device: torch.device) -> torch.Tensor:
    """
    Precompute the (ny, nx, n_det, 2) LUT of (t_idx_floor, alpha) for linear interpolation over time
    used by the Delay-and-Sum (DAS) beamformer.
    - t_idx_floor: integer index of the left temporal sample
    - alpha: fractional part for linear interpolation: s(t0+alpha*dt) between floor and floor+1
    Notes:
      * We assume 2D propagation; for each pixel and detector, compute distance r and time t = r/c.
      * Then map to temporal sample index k = (t - t0) / dt.
      * Out-of-range indices will be masked at runtime.
    """
    # Pixel grid in meters
    xs = geom.x0_m + np.arange(geom.nx, dtype=np.float32) * geom.dx_m
    ys = geom.y0_m + np.arange(geom.ny, dtype=np.float32) * geom.dy_m
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)

    det_x = geom.det_x.astype(np.float32)          # (n_det,)
    det_y = geom.det_y.astype(np.float32)          # (n_det,)

    # Expand dims for broadcasting
    Xe = X[..., None]                               # (ny, nx, 1)
    Ye = Y[..., None]                               # (ny, nx, 1)
    Dx = det_x[None, None, :]                       # (1, 1, n_det)
    Dy = det_y[None, None, :]                       # (1, 1, n_det)

    # Euclidean distance from each pixel to each detector
    R = np.sqrt((Xe - Dx) ** 2 + (Ye - Dy) ** 2)    # (ny, nx, n_det)
    T = R / geom.c_m_s                              # time-of-flight [s]

    # Convert time to fractional temporal index
    K = (T - geom.t0_s) / geom.dt_s                 # (ny, nx, n_det)
    K_floor = np.floor(K)
    Alpha = (K - K_floor).astype(np.float32)
    K_floor = K_floor.astype(np.float32)

    # Pack LUT (t_idx_floor, alpha)
    lut = np.stack([K_floor, Alpha], axis=-1).astype(np.float32)  # (ny, nx, n_det, 2)

    # To torch (ensure float32 dtype)
    lut_t = torch.from_numpy(lut).to(device=device, dtype=torch.float32)
    return lut_t


class DelayAndSumLinear(nn.Module):
    """
    Differentiable Delay-and-Sum beamformer using a precomputed LUT for a linear array.
    Input:  sinogram S with shape [B, 1, n_det, n_t]
    Output: image I with shape [B, 1, ny, nx]
    Notes:
      * We linearly interpolate along the temporal dimension using (floor, alpha).
      * Out-of-range temporal indices are masked to zero contribution.
      * Summation over detectors approximates classical DAS (unweighted). You can add apodization later.
    """
    def __init__(
        self,
        geom: LinearProbeGeom,
        lut: torch.Tensor,
        trainable_apodization: bool = False,
    ):
        super().__init__()
        self.geom = geom
        self.trainable_apodization = trainable_apodization
        lut = lut.to(dtype=torch.float32)
        self.register_buffer("lut", lut)  # (ny, nx, n_det, 2)

        # Pre-compute interpolation helpers (k0, alpha, valid mask)
        k_floor = torch.floor(lut[..., 0])
        valid = (k_floor >= 0) & (k_floor < (geom.n_t - 1))
        max_idx = max(geom.n_t - 2, 0)
        k0 = torch.clamp(k_floor, 0, max_idx).to(dtype=torch.long)
        alpha = lut[..., 1].to(dtype=torch.float32)
        self.register_buffer("k0", k0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("valid", valid)

        # Optional apodization per-detector (Hanning). Shape (n_det,)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(geom.n_det) / max(geom.n_det - 1, 1))
        apod = torch.from_numpy(win.astype(np.float32))
        if trainable_apodization:
            self.apod = nn.Parameter(apod)
        else:
            self.register_buffer("apod", apod)

    def get_apodization(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return the apodization weights in the requested dtype/device."""
        apod = self.apod
        if self.trainable_apodization:
            apod = apod.abs()
        apod = apod.to(device=device, dtype=dtype)
        if self.trainable_apodization:
            apod = apod.clamp_min(torch.finfo(apod.dtype).tiny)
        return apod

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """
        sino: [B, 1, n_det, n_t]
        returns: [B, 1, ny, nx]
        """
        B, C, n_det, n_t = sino.shape
        assert C == 1, "Expect sinogram with a single channel."
        assert n_det == self.geom.n_det and n_t == self.geom.n_t, "Sinogram dims mismatch geometry."

        # Reuse pre-computed interpolation helpers
        k0 = self.k0
        k1 = k0 + 1
        alpha = self.alpha
        if alpha.dtype != sino.dtype:
            alpha = alpha.to(dtype=sino.dtype)
        valid = self.valid

        # Gather s(k0) and s(k1) from sino: need shapes aligned to (B, n_det, ny, nx)
        # We'll index per-detector and per-time, then interpolate and sum over detectors.
        # Reshape for easier gather: [B, n_det, n_t]
        S = sino[:, 0, :, :]  # [B, n_det, n_t]

        # Prepare output accumulator
        out = torch.zeros((B, self.geom.ny, self.geom.nx), device=sino.device, dtype=sino.dtype)

        # Vectorized gather: expand dims to (B, ny, nx, n_det)
        # We need to index S[b, det, t_idx] with t_idx = k0/k1[..., det]
        k0e = k0.unsqueeze(0).expand(B, -1, -1, -1)        # (B, ny, nx, n_det)
        k1e = k1.unsqueeze(0).expand(B, -1, -1, -1)
        a   = alpha.unsqueeze(0).expand(B, -1, -1, -1)     # (B, ny, nx, n_det)
        ve  = valid.unsqueeze(0).expand(B, -1, -1, -1)

        # For S gather, bring dims to enable torch.gather along time dim:
        # We will index time dim last, so create S_det_time per-batch per-detector per-time:
        # We'll gather per-detector time then select detector via advanced indexing.
        # A practical route: loop over detector dimension to keep memory reasonable & simple.
        # (n_det is typically O(64-256), loop is acceptable and safe in FP32)
        apod = self.get_apodization(sino.dtype, sino.device)
        for d in range(self.geom.n_det):
            # time indices for this detector across the image grid
            k0_d = k0e[..., d]  # (B, ny, nx)
            k1_d = k1e[..., d]
            a_d  = a[..., d]
            v_d  = ve[..., d]

            # Gather S[:, d, k] -> (B, ny, nx)
            s0 = S[:, d, :].gather(dim=1, index=k0_d.reshape(B, -1)).reshape(B, self.geom.ny, self.geom.nx)
            s1 = S[:, d, :].gather(dim=1, index=k1_d.reshape(B, -1)).reshape(B, self.geom.ny, self.geom.nx)

            # Linear interpolation
            sk = (1.0 - a_d) * s0 + a_d * s1

            # Mask out-of-range, apply apodization, and accumulate
            sk = sk * v_d.to(sk.dtype)
            out += apod[d] * sk

        # Normalize by number of detectors (and apodization sum) to keep scale stable
        norm = torch.clamp(apod.sum(), min=torch.finfo(apod.dtype).tiny)
        out = out / norm

        return out.unsqueeze(1)  # [B, 1, ny, nx]


class FkMigrationLinear(nn.Module):
    """Frequency-wavenumber migration for linear probe acquisitions."""

    def __init__(
        self,
        geom: LinearProbeGeom,
        trainable_apodization: bool = False,
        per_channel_apodization: bool = False,
    ):
        super().__init__()
        self.geom = geom
        self.trainable_apodization = trainable_apodization
        self.per_channel_apodization = per_channel_apodization

        # Temporal frequency grid (angular frequency ω)
        freq = torch.fft.rfftfreq(geom.n_t, d=geom.dt_s)
        omega = (2.0 * math.pi * freq).to(dtype=torch.float32)
        self.register_buffer("omega", omega)

        # Weights for positive frequencies (double interior terms)
        weight = torch.ones_like(omega)
        if omega.numel() > 2:
            weight[1:-1] = 2.0
        if geom.n_t % 2 == 0 and omega.numel() > 1:
            weight[-1] = 1.0
        self.register_buffer("freq_weight", weight)
        self.register_buffer("freq_weight_sum", weight.sum())

        # Lateral wavenumbers k_x
        kx = torch.fft.fftfreq(geom.n_det, d=geom.pitch_m)
        kx = (2.0 * math.pi * kx).to(dtype=torch.float32)
        self.register_buffer("kx", kx)

        # Pixel coordinates in meters
        xs = geom.x0_m + torch.arange(geom.nx, dtype=torch.float32) * geom.dx_m
        ys = geom.y0_m + torch.arange(geom.ny, dtype=torch.float32) * geom.dy_m
        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)

        # Optional apodization window (Hanning)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(geom.n_det) / max(geom.n_det - 1, 1))
        apod = torch.from_numpy(win.astype(np.float32))
        if per_channel_apodization:
            apod = apod.unsqueeze(0)
        if trainable_apodization:
            self.apod = nn.Parameter(apod)
        else:
            self.register_buffer("apod", apod)

    def get_apodization(self, dtype: torch.dtype, device: torch.device, channels: int) -> torch.Tensor:
        """Return detector apodization replicated per channel if requested."""
        apod = self.apod
        if self.trainable_apodization:
            apod = apod.abs()
        apod = apod.to(device=device, dtype=dtype)

        if self.per_channel_apodization:
            if apod.dim() == 1:
                apod = apod.unsqueeze(0)
            if apod.shape[0] == 1 and channels > 1:
                apod = apod.expand(channels, -1)
            elif apod.shape[0] != channels:
                base = apod.mean(dim=0, keepdim=True)
                apod = base.expand(channels, -1)
        else:
            if apod.dim() == 2:
                apod = apod[0]
            apod = apod.view(1, -1).expand(channels, -1)

        if self.trainable_apodization:
            apod = apod.clamp_min(torch.finfo(apod.dtype).tiny)
        return apod

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """Apply f-k migration to a sinogram of shape [B, C, n_det, n_t]."""
        B, C, n_det, n_t = sino.shape
        assert n_det == self.geom.n_det and n_t == self.geom.n_t, "Sinogram dims mismatch geometry."

        dtype = sino.dtype
        device = sino.device
        eps = torch.finfo(dtype).eps

        # Normalize each sinogram to reduce dynamic range differences
        mean = sino.mean(dim=(-1, -2), keepdim=True)
        sino_norm = sino - mean
        var = sino_norm.pow(2).mean(dim=(-1, -2), keepdim=True)
        std = torch.sqrt(var + eps)
        sino_norm = sino_norm / std

        apod = self.get_apodization(dtype, device, C)
        apod_view = apod.unsqueeze(0).unsqueeze(-1)  # [1, C, n_det, 1]
        sino_weighted = sino_norm * apod_view

        # Frequency transforms
        spec_t = torch.fft.rfft(sino_weighted, dim=-1)
        spec_fk = torch.fft.fft(spec_t, dim=2)

        real_dtype = spec_fk.real.dtype
        omega = self.omega.to(device=device, dtype=real_dtype)
        kx = self.kx.to(device=device, dtype=real_dtype)

        omega_term = (omega / self.geom.c_m_s) ** 2  # (n_freq,)
        kx_sq = kx ** 2  # (n_det,)

        kz_sq = omega_term.view(1, 1, 1, -1) - kx_sq.view(1, 1, -1, 1)
        kz_sq = torch.clamp(kz_sq, min=0.0)
        kz = torch.sqrt(kz_sq)
        prop_mask = (kz_sq > 0).to(spec_fk.dtype)

        freq_weight = self.freq_weight.to(device=device, dtype=real_dtype)
        freq_weight_sum = torch.clamp(self.freq_weight_sum.to(device=device, dtype=dtype), min=eps)
        freq_weight = freq_weight.view(1, 1, 1, -1).to(dtype=spec_fk.dtype)

        spec_fk = spec_fk * prop_mask

        x_phase_coords = (self.x_coords - self.geom.array_x0_m).to(device=device, dtype=real_dtype)
        y_phase_coords = (self.y_coords - self.geom.array_y_m).to(device=device, dtype=real_dtype)
        phase_x = torch.exp(1j * (kx.view(-1, 1) * x_phase_coords.view(1, -1)))
        phase_x = phase_x.to(dtype=spec_fk.dtype)

        kz = kz.to(dtype=spec_fk.real.dtype)
        lines: List[torch.Tensor] = []
        for y_val in y_phase_coords:
            phase_y = torch.exp(1j * (kz * y_val))
            phase_y = phase_y.to(dtype=spec_fk.dtype)
            weighted = spec_fk * phase_y * freq_weight
            band = weighted.sum(dim=-1)  # [B, C, n_det]
            band = band.reshape(B * C, self.geom.n_det)
            line = torch.matmul(band, phase_x)
            line = line.view(B, C, self.geom.nx)
            lines.append(line)

        img_complex = torch.stack(lines, dim=2)  # [B, C, ny, nx]
        img_mag = img_complex.abs().to(dtype)

        apod_sum = torch.clamp(apod.sum(dim=-1, keepdim=True), min=eps)
        norm = apod_sum.view(1, C, 1, 1) * freq_weight_sum.reshape(1, 1, 1, 1)
        img_mag = img_mag / norm

        return img_mag


class ForwardProjectionLinear(nn.Module):
    """
    Differentiable forward projection that reuses the LUT/apodization from
    :class:`DelayAndSumLinear` for a linear array acquisition geometry.
    Input:  image I with shape [B, 1, ny, nx]
    Output: sinogram S with shape [B, 1, n_det, n_t]
    """

    def __init__(self, das: DelayAndSumLinear):
        super().__init__()
        self.geom = das.geom
        # Keep an internal reference without registering the module twice
        object.__setattr__(self, "_das", das)

    @property
    def das(self) -> DelayAndSumLinear:
        return self._das

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 1, ny, nx]
        returns: [B, 1, n_det, n_t]
        """
        B, C, ny, nx = img.shape
        assert C == 1, "Expect image with a single channel."
        assert ny == self.geom.ny and nx == self.geom.nx, "Image dims mismatch geometry."

        k0 = self.das.k0
        if k0.device != img.device:
            k0 = k0.to(device=img.device)
        k1 = k0 + 1
        alpha = self.das.alpha.to(device=img.device, dtype=img.dtype)
        valid = self.das.valid
        if valid.device != img.device:
            valid = valid.to(device=img.device)

        img_flat = img[:, 0, :, :].reshape(B, -1)  # [B, ny*nx]
        sino = torch.zeros((B, self.geom.n_det, self.geom.n_t), device=img.device, dtype=img.dtype)

        apod = self.das.get_apodization(img.dtype, img.device)

        for d in range(self.geom.n_det):
            k0_d = k0[..., d].reshape(-1)
            k1_d = k1[..., d].reshape(-1)
            alpha_d = alpha[..., d].reshape(-1)
            valid_d = valid[..., d].reshape(-1)

            mask = valid_d.to(dtype=img.dtype)
            w0 = (1.0 - alpha_d) * mask
            w1 = alpha_d * mask

            idx0 = k0_d.unsqueeze(0).expand(B, -1)
            idx1 = k1_d.unsqueeze(0).expand(B, -1)
            src0 = img_flat * w0.unsqueeze(0)
            src1 = img_flat * w1.unsqueeze(0)

            sino[:, d, :].scatter_add_(dim=1, index=idx0, src=src0 * apod[d])
            sino[:, d, :].scatter_add_(dim=1, index=idx1, src=src1 * apod[d])

        norm = torch.clamp(apod.sum(), min=torch.finfo(apod.dtype).tiny)
        sino = sino / norm

        return sino.unsqueeze(1)

# ================================
# Lightweight ViT-like Refiner
# ================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_ch=1, embed_dim=256, patch=16, stride=None):
        super().__init__()
        self.patch = patch
        self.patch_size = _pair(patch)
        self.stride = patch if stride is None else stride
        self.stride_size = _pair(self.stride)
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride_size,
        )

    def compute_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        ph, pw = self.patch_size
        sh, sw = self.stride_size
        Hp = max((H - ph) // sh + 1, 0)
        Wp = max((W - pw) // sw + 1, 0)
        return Hp, Wp

    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, D]
        B, C, H, W = x.shape
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, D]
        return x, (Hp, Wp), (H, W), self.stride_size

class TransformerEncoder(nn.Module):
    """A small stack of standard Transformer encoder blocks."""
    def __init__(self, dim=256, depth=6, heads=8, mlp_ratio=4.0, p_drop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict({
                    "ln1": nn.LayerNorm(dim),
                    "attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=p_drop, batch_first=True),
                    "ln2": nn.LayerNorm(dim),
                    "mlp": nn.Sequential(
                        nn.Linear(dim, int(dim*mlp_ratio)),
                        nn.GELU(),
                        nn.Linear(int(dim*mlp_ratio), dim)
                    )
                })
            )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        # x: [B, N, D]
        for blk in self.blocks:
            # Self-attention with residual
            y = blk["ln1"](x)
            y, _ = blk["attn"](y, y, y, need_weights=False)
            x = x + self.drop(y)
            # MLP with residual
            y = blk["ln2"](x)
            y = blk["mlp"](y)
            x = x + self.drop(y)
        return x


class PatchDecoder(nn.Module):
    """Decode token features on the patch grid with local convolutions."""

    def __init__(self, embed_dim: int, patch_area: int, num_blocks: int = 3):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("PatchDecoder requires at least one convolutional block")

        layers: List[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv2d(embed_dim, patch_area, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)
        x = x + residual
        x = self.proj(x)
        return x


class LocalFusionBlock(nn.Module):
    """Fuse overlapping patches on the full-resolution image with local smoothing."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class ViTRefiner(nn.Module):
    """
    Simple ViT-like refiner that denoises/refines the Delay-and-Sum (DAS) image.
    Input:  [B, 1, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(
        self,
        in_ch=1,
        embed_dim=256,
        patch=16,
        stride=None,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        p_drop=0.1,
    ):
        super().__init__()
        self.patch = patch
        self.stride = patch if stride is None else stride
        self.patch_size = _pair(self.patch)
        self.stride_size = _pair(self.stride)
        self.embed = PatchEmbed(in_ch, embed_dim, patch, stride=self.stride)
        self.embed_dim = embed_dim
        self.pos_embed: Optional[nn.Parameter] = None  # created on first forward
        self.grid_size: Optional[Tuple[int, int]] = None
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio, p_drop)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.token_decoder = PatchDecoder(embed_dim, self.patch_area, num_blocks=3)
        fusion_hidden = max(embed_dim // 8, 16)
        self.local_fusion = LocalFusionBlock(in_channels=1, hidden_channels=fusion_hidden)
        self.register_buffer("fold_weight", torch.empty(0), persistent=False)
        self._fold_weight_shape: Optional[Tuple[int, int]] = None
        self._fold_weight_stride: Optional[Tuple[int, int]] = None

    def _compute_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        return self.embed.compute_grid_size(H, W)

    def _build_pos_embed(
        self,
        H: int,
        W: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        Hp, Wp = self._compute_grid_size(H, W)
        pe = torch.zeros(1, Hp * Wp, dim, device=device, dtype=dtype)
        pe = nn.Parameter(pe)
        nn.init.trunc_normal_(pe, std=0.02)
        self.pos_embed = pe
        self.grid_size = (Hp, Wp)

    def _get_fold_weight(
        self,
        shape: Tuple[int, int],
        stride: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = self.fold_weight
        if (
            weight.numel() == 0
            or self._fold_weight_shape != shape
            or self._fold_weight_stride != stride
        ):
            ones = torch.ones(1, 1, shape[0], shape[1], device=device, dtype=dtype)
            weight = F.fold(
                F.unfold(ones, kernel_size=self.patch_size, stride=stride),
                output_size=shape,
                kernel_size=self.patch_size,
                stride=stride,
            )
            weight = weight.to(device=device, dtype=dtype)
            self.fold_weight = weight
            self._fold_weight_shape = shape
            self._fold_weight_stride = stride
        elif weight.device != device or weight.dtype != dtype:
            weight = weight.to(device=device, dtype=dtype)
            self.fold_weight = weight
        return weight

    def forward(self, x):
        # x: [B, 1, H, W]
        z, (Hp, Wp), (H, W), stride = self.embed(x)  # z: [B, N, D]
        stride_hw = tuple(stride)
        expected_grid = self._compute_grid_size(H, W)
        if expected_grid != (Hp, Wp):
            Hp, Wp = expected_grid
        B, N, D = z.shape
        if Hp * Wp != N:
            raise ValueError(
                f"Token count ({N}) does not match grid size ({Hp}x{Wp})."
            )
        if (
            self.pos_embed is None
            or self.pos_embed.shape[1] != N
            or self.pos_embed.shape[2] != D
            or self.grid_size != (Hp, Wp)
        ):
            self._build_pos_embed(H, W, D, x.device, dtype=z.dtype)
        z = z + self.pos_embed
        z = self.encoder(z)                        # [B, N, D]
        z = z.transpose(1, 2).reshape(B, D, Hp, Wp)
        pixels = self.token_decoder(z)             # [B, patch_area, Hp, Wp]
        pixels = pixels.reshape(B, self.patch_area, Hp * Wp)
        img = F.fold(
            pixels,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=stride_hw,
        )
        norm = self._get_fold_weight((H, W), stride_hw, x.device, x.dtype)
        img = img / norm.clamp_min(1.0)
        img = self.local_fusion(img)
        if img.shape == x.shape:
            img = img + x
        return img


def adapt_vitrefiner_state_dict(state_dict: dict) -> Tuple[dict, bool]:
    """Upgrade legacy ViTRefiner checkpoints to the new convolutional decoder."""

    new_state = dict(state_dict)
    converted = False

    suffix_weight = "proj_out.weight"
    suffix_bias = "proj_out.bias"

    for key in list(state_dict.keys()):
        if key.endswith(suffix_weight):
            prefix = key[: -len(suffix_weight)]
            bias_key = prefix + suffix_bias
            weight = new_state.pop(key)
            bias = new_state.pop(bias_key, None)

            conv_weight = weight.new_zeros((weight.shape[0], weight.shape[1], 3, 3))
            conv_weight[:, :, 1, 1] = weight

            new_state[prefix + "token_decoder.proj.weight"] = conv_weight
            if bias is not None:
                new_state[prefix + "token_decoder.proj.bias"] = bias
            else:
                new_state[prefix + "token_decoder.proj.bias"] = weight.new_zeros(weight.shape[0])
            converted = True
        elif key.endswith("out_conv.weight") or key.endswith("out_conv.bias"):
            new_state.pop(key)
            converted = True

    return new_state, converted


# ================================
# Full Model: Delay-and-Sum (fixed) + Transformer (trainable)
# ================================

class DelayAndSumTransformer(nn.Module):
    def __init__(self, das: DelayAndSumLinear, vit: ViTRefiner, freeze_das: bool = True):
        super().__init__()
        self.das = das
        self.vit = vit
        if freeze_das:
            for p in self.das.parameters():
                p.requires_grad = False

    def forward(self, sino: torch.Tensor):
        das_img = self.das(sino)   # [B, 1, H, W]
        out = self.vit(das_img)    # [B, 1, H, W]
        intermediates = [out]
        return out, das_img, intermediates


class UnrolledDelayAndSumTransformer(nn.Module):
    def __init__(
        self,
        das_module: DelayAndSumLinear,
        forward_module: ForwardProjectionLinear,
        vit_module: ViTRefiner,
        num_steps: int,
        data_consistency_weight: float = 1.0,
        learnable_data_consistency_weight: bool = False,
        freeze_das: bool = False,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1 for UnrolledDelayAndSumTransformer")
        self.das = das_module
        self.forward_projector = forward_module
        self.vit = vit_module
        self.num_steps = int(num_steps)
        weight = torch.tensor(float(data_consistency_weight), dtype=torch.float32)
        if learnable_data_consistency_weight:
            self.data_consistency_weight = nn.Parameter(weight)
        else:
            self.register_buffer("data_consistency_weight", weight)
        if freeze_das:
            for p in self.das.parameters():
                p.requires_grad = False

    def forward(self, sino: torch.Tensor):
        x0 = self.das(sino)
        xi = x0
        intermediates: List[torch.Tensor] = []

        weight = self.data_consistency_weight.view(1, 1, 1, 1)

        for _ in range(self.num_steps):
            sino_est = self.forward_projector(xi)
            sino_residual = sino - sino_est
            correction = self.das(sino_residual)
            xi = xi + weight * correction
            xi = self.vit(xi)
            intermediates.append(xi)

        return xi, x0, intermediates


def run_inference_steps(
    model: nn.Module,
    sinogram: torch.Tensor,
    cfg: "TrainConfig",
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
    """
    Return normalized sinogram, Delay-and-Sum (DAS) image, final output and the full per-step
    sequence.

    ``iter_imgs`` always includes the DAS image as step 0 and, when available, all
    subsequent iterations up to the final reconstruction. When a model does not
    expose intermediate steps the list may contain only the DAS result and the final
    prediction (or be ``None`` when unavailable).
    """
    if device is None:
        device = next(model.parameters()).device

    if sinogram.dim() == 2:
        sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    elif sinogram.dim() == 3:
        if sinogram.shape[0] != 1 and sinogram.shape[1] != 1:
            sinogram = sinogram.unsqueeze(1)
        elif sinogram.shape[0] == 1:
            sinogram = sinogram.unsqueeze(0)
    elif sinogram.dim() != 4:
        raise ValueError(f"Unexpected sinogram shape: {tuple(sinogram.shape)}")

    dtype = next(model.parameters()).dtype
    sino = sinogram.to(device=device, dtype=dtype)
    sino_norm = minmax_scale(sino, cfg.sino_min, cfg.sino_max) if normalize else sino

    model.eval()
    with torch.no_grad():
        pred, das_img, intermediates = model(sino_norm)

    iter_sequence: List[torch.Tensor] = []
    if das_img is not None:
        iter_sequence.append(das_img)

    if intermediates is not None:
        if isinstance(intermediates, (list, tuple)):
            iter_sequence.extend(intermediates)
        else:
            iter_sequence.append(intermediates)

    if pred is not None and (not iter_sequence or iter_sequence[-1] is not pred):
        iter_sequence.append(pred)

    iter_imgs = [step.detach().cpu() for step in iter_sequence] if iter_sequence else None

    return (
        sino_norm.detach().cpu(),
        das_img.detach().cpu(),
        pred.detach().cpu(),
        iter_imgs,
    )

# ================================
# Metrics (PSNR/SSIM/L1)
# ================================

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Compute PSNR per-image assuming inputs in [0,1] or similar scale."""
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))

def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    C1=0.01**2,
    C2=0.03**2,
    window=11,
    mask: Optional[torch.Tensor] = None,
):
    """
    Very lightweight SSIM approximation on single-channel images.
    Assumes inputs normalized to [0,1]. Not a full reference implementation, but good for monitoring.

    If ``mask`` is provided it must broadcast to the image shape ``[B, 1, H, W]`` and only
    masked pixels contribute to the spatial average. When the mask is empty the global
    average is returned as a safe fallback.
    """
    # Gaussian window approximation with uniform kernel for simplicity
    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, 1, pad)
    mu_y = F.avg_pool2d(target, window, 1, pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred * pred, window, 1, pad) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, window, 1, pad) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window, 1, pad) - mu_xy

    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    ssim_flat = ssim_map.flatten(1)

    if mask is None:
        # Mean over spatial dims per-batch
        return ssim_flat.mean(dim=1)

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape != ssim_map.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} incompatible with SSIM map shape {tuple(ssim_map.shape)}"
        )

    mask = mask.to(ssim_map.dtype)
    mask_flat = mask.flatten(1)
    masked_sum = (ssim_flat * mask_flat).sum(dim=1)
    mask_norm = mask_flat.sum(dim=1)
    global_mean = ssim_flat.mean(dim=1)
    masked_mean = masked_sum / mask_norm.clamp_min(1.0)
    return torch.where(mask_norm > 0, masked_mean, global_mean)

# ================================
# Side-by-side saver
# ================================

def save_side_by_side(
    pred: torch.Tensor,
    gt: torch.Tensor,
    das: torch.Tensor,
    out_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    iter_steps: Optional[List[Tuple[int, torch.Tensor]]] = None,
):
    """
    Save a side-by-side image [DAS | (optional intermediate steps) | Pred | GT].

    Expects tensors in shape [1, H, W] and roughly [0,1] range.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def to_uint8(img: torch.Tensor) -> Image.Image:
        x = img.detach().cpu()
        if x.dim() == 3 and x.shape[0] == 1:
            x = x[0]
        elif x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 1:
            x = x[0, 0]
        else:
            x = x.squeeze()

        x = x.clamp(0, 1)
        if vmin is not None or vmax is not None:
            lo = 0.0 if vmin is None else vmin
            hi = 1.0 if vmax is None else vmax
            x = (x - lo) / max(hi - lo, 1e-6)
            x = x.clamp(0, 1)
        x = (x * 255.0).round().byte().numpy()  # [H,W]
        return Image.fromarray(x, mode="L")

    panels: List[Image.Image] = []

    panels.append(to_uint8(das))

    if iter_steps:
        seen = set()
        for idx, tensor in sorted(iter_steps, key=lambda item: item[0]):
            if idx in seen:
                continue
            seen.add(idx)
            panels.append(to_uint8(tensor))

    panels.append(to_uint8(pred))
    panels.append(to_uint8(gt))

    width = sum(im.width for im in panels)
    height = max(im.height for im in panels)
    canvas = Image.new("L", (width, height))
    xoff = 0
    for im in panels:
        canvas.paste(im, (xoff, 0))
        xoff += im.width
    canvas.save(out_path)

# ================================
# Training / Validation (FP32 ONLY)
# ================================

def compute_intensity_weights(img: torch.Tensor, alpha: float, threshold: Optional[float]) -> torch.Tensor:
    """Build per-pixel weights from the ground-truth intensities."""
    weights = torch.ones_like(img)
    if alpha == 0.0:
        return weights
    if threshold is None:
        weights = weights + alpha * img
    else:
        weights = weights + alpha * (img > threshold).to(img.dtype)
    return weights


def build_bright_mask(
    img: torch.Tensor,
    threshold: Optional[float],
    dilation: int = 0,
) -> torch.Tensor:
    """Create a binary mask highlighting bright regions in ``img``."""
    if threshold is None:
        mask = torch.ones_like(img)
    else:
        mask = (img > threshold).to(img.dtype)

    if dilation and dilation > 0:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
        mask = (mask > 0).to(img.dtype)

    return mask


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch,
    save_dir: Optional[str] = None,
    max_save: int = 8,
    intermediate_indices: Optional[List[int]] = None,
    weight_alpha: float = 0.0,
    weight_threshold: Optional[float] = None,
    ssim_mask_threshold: Optional[float] = None,
    ssim_mask_dilation: int = 0,
    use_tqdm: bool = True,
):
    model.eval()
    agg = {'psnr': 0.0, 'ssim': 0.0, 'l1': 0.0, 'weighted_l1': 0.0}
    compute_masked_ssim = ssim_mask_threshold is not None
    if compute_masked_ssim:
        agg['masked_ssim'] = 0.0
    n = 0
    saved = 0
    with torch.no_grad():  # NOTE: no autocast anywhere (pure FP32)
        progress = tqdm(loader, desc="Val") if use_tqdm else None
        iterator = progress if progress is not None else loader
        for i, (sino, img) in enumerate(iterator):
            sino = sino.to(device, non_blocking=True)
            img  = img.to(device, non_blocking=True)

            pred, das, intermediates = model(sino)           # forward in FP32
            iter_sequence: List[torch.Tensor] = [das]
            if intermediates is not None:
                if isinstance(intermediates, (list, tuple)):
                    iter_sequence.extend(intermediates)
                else:
                    iter_sequence.append(intermediates)
            if pred is not None and (not iter_sequence or iter_sequence[-1] is not pred):
                iter_sequence.append(pred)
            # Metrics per-sample
            batch_psnr = psnr(pred, img).mean()
            batch_ssim = ssim(pred, img).mean()
            batch_l1   = F.l1_loss(pred, img)
            weights = compute_intensity_weights(img, weight_alpha, weight_threshold)
            batch_weighted_l1 = torch.mean(weights * torch.abs(pred - img))

            if compute_masked_ssim:
                mask = build_bright_mask(img, ssim_mask_threshold, dilation=ssim_mask_dilation)
                batch_masked_ssim = ssim(pred, img, mask=mask).mean()

            bs = sino.size(0)
            n += bs
            agg['psnr'] += batch_psnr.item() * bs
            agg['ssim'] += batch_ssim.item() * bs
            agg['l1']   += batch_l1.item() * bs
            agg['weighted_l1'] += batch_weighted_l1.item() * bs
            if compute_masked_ssim:
                agg['masked_ssim'] += batch_masked_ssim.item() * bs

            if progress is not None:
                denom = max(n, 1)
                postfix = {
                    'psnr': f"{agg['psnr'] / denom:.4f}",
                    'ssim': f"{agg['ssim'] / denom:.4f}",
                    'l1': f"{agg['l1'] / denom:.4f}",
                    'w_l1': f"{agg['weighted_l1'] / denom:.4f}",
                }
                if compute_masked_ssim:
                    postfix['masked_ssim'] = f"{agg['masked_ssim'] / denom:.4f}"
                progress.set_postfix(postfix)

            # Save a few side-by-side for inspection
            if save_dir is not None and saved < max_save:
                for b in range(min(bs, max_save - saved)):
                    out_path = os.path.join(save_dir, f"val_epoch_{epoch}_{i:04d}_più corta{b:02d}.png")
                    debug_steps: Optional[List[Tuple[int, torch.Tensor]]] = None
                    if intermediate_indices and iter_sequence:
                        total_steps = len(iter_sequence)
                        selected: List[Tuple[int, torch.Tensor]] = []
                        seen_steps = set()
                        for idx in intermediate_indices:
                            actual_idx = idx if idx >= 0 else total_steps + idx
                            if actual_idx < 0 or actual_idx >= total_steps:
                                continue
                            if actual_idx in seen_steps:
                                continue
                            seen_steps.add(actual_idx)
                            if actual_idx == 0 or actual_idx == total_steps - 1:
                                continue
                            selected.append((actual_idx, iter_sequence[actual_idx][b]))
                        if selected:
                            selected.sort(key=lambda item: item[0])
                            debug_steps = selected
                    save_side_by_side(
                        pred[b],
                        img[b],
                        das[b],
                        out_path,
                        iter_steps=debug_steps,
                    )
                    saved += 1

        if progress is not None:
            progress.close()

    for k in agg:
        agg[k] /= max(n, 1)
    return agg


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device,
                    clip_grad: float = 1.0, weight_alpha: float = 0.0,
                    weight_threshold: Optional[float] = None, use_tqdm: bool = True):
    model.train()
    agg = {'psnr': 0.0, 'ssim': 0.0, 'l1': 0.0, 'weighted_l1': 0.0}
    n = 0
    progress = tqdm(loader, desc="Train") if use_tqdm else None
    iterator = progress if progress is not None else loader
    for i, (sino, img) in enumerate(iterator):
        sino = sino.to(device, non_blocking=True)
        img  = img.to(device, non_blocking=True)

        pred, das_img, intermediates = model(sino)                    # forward in FP32
        weights = compute_intensity_weights(img, weight_alpha, weight_threshold)
        loss_weighted_l1 = torch.mean(weights * torch.abs(pred - img))

        optimizer.zero_grad(set_to_none=True)
        loss_weighted_l1.backward()               # backprop in FP32
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                clip_grad,
            )
        optimizer.step()

        # metrics on-the-fly
        with torch.no_grad():
            bs = sino.size(0)
            n += bs
            agg['weighted_l1'] += loss_weighted_l1.item() * bs
            agg['l1']   += F.l1_loss(pred, img).item() * bs
            agg['psnr'] += psnr(pred, img).mean().item() * bs
            agg['ssim'] += ssim(pred, img).mean().item() * bs

            if progress is not None:
                denom = max(n, 1)
                progress.set_postfix({
                    'psnr': f"{agg['psnr'] / denom:.4f}",
                    'ssim': f"{agg['ssim'] / denom:.4f}",
                    'l1': f"{agg['l1'] / denom:.4f}",
                    'w_l1': f"{agg['weighted_l1'] / denom:.4f}",
                })

    for k in agg:
        agg[k] /= max(n, 1)
    if progress is not None:
        progress.close()
    return agg

@torch.no_grad()
def compute_global_minmax_from_loader(loader, get_sino, get_img):
    """Compute per-domain global min/max across the whole training loader."""
    smin, smax = np.inf, -np.inf
    imin, imax = np.inf, -np.inf
    for batch in loader:
        sino = get_sino(batch)  # torch tensor [B, 1, n_det, n_t] or similar
        img  = get_img(batch)   # torch tensor [B, 1, H, W]
        sino_np = sino.detach().cpu().numpy()
        img_np  = img.detach().cpu().numpy()
        smin = min(smin, np.nanmin(sino_np)); smax = max(smax, np.nanmax(sino_np))
        imin = min(imin, np.nanmin(img_np));  imax = max(imax, np.nanmax(img_np))
    return {"sino": {"min": float(smin), "max": float(smax)},
            "img":  {"min": float(imin), "max": float(imax)}}

def save_stats_json(stats: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def load_stats_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# ================================
# Orchestrator
# ================================
@dataclass
class TrainConfig:
    # Geometry / sampling
    n_det: int = 128
    pitch_m: float = 0.0003            # Spaziatura (center-to-center) tra elementi della sonda, in metri.
    t0_s: float = 0.0                  # Tempo di inizio dell’acquisizione, in secondi. Può servire se c’è un ritardo (es. pre-trigger).
    dt_s: float = 1/31.25e6            # Passo di campionamento temporale, in secondi. 1/(frquenza di campionamento Hz)
    n_t: int = 1640                    # Numero di campioni temporali per ciascun detector.
    c_m_s: float = 1540.0              # Velocità del suono nei tessuti molli (m/s). Usata per convertire tempi di volo in distanze.
    x0_m: float = -0.019               # Coordinata x del bordo sinistro dell’immagine (in metri). ~ - (nx * dx)/2  if centered
    y0_m: float = 0.0                  # Coordinata y (profondità) del bordo superiore dell’immagine.
    dx_m: float = 0.00015              # Spaziatura tra pixel sull’asse x.
    dy_m: float = 0.00015              # Spaziatura tra pixel sull’asse y.
    nx: int = 256
    ny: int = 256
    array_x0_m: float = -0.019         # align first element with left image border (example)
    array_y_m: float = 0.0
    wavelength: int = 800
    trainable_apodization: bool = True

    # ViT refiner
    vit_patch: int = 16
    vit_stride: Optional[int] = None

    # Training
    epochs: int = 50
    batch_size: int = 4
    lr: float = 2e-4
    num_workers: int = 4
    clip_grad: float = 1.0
    use_tqdm: bool = True
    weight_alpha: float = 1.0
    weight_threshold: Optional[float] = 0.5
    ssim_mask_threshold: Optional[float] = 0.5
    ssim_mask_dilation: int = 0

    # Model variants
    model_variant: str = "unrolled"
    unroll_steps: int = 7
    data_consistency_weight: float = 1.0
    learnable_data_consistency_weight: bool = True

    # --- Global scaling (per-domain) ---
    # If any of these is None, stats will be computed from the training set.
    sino_min: float = -11.0322
    sino_max: float = 12.5394
    img_min: float = 0.0
    img_max: float = 316.9658

    # Paths
    work_dir: str = "./runs/das_transformer_fp32_7_unroll"
    data_root: str = "E:/Scardigno/datasets_transformer_proj"
    sino_dir: str = "Forearm2000_hdf5/train_val_tst"
    recs_dir: str = "Forearm2000_recs/L1_Shearlet"
    save_val_images: bool = True
    max_val_images: int = 1
    val_intermediate_indices: Optional[List[int]] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # steps (0-based, allow negatives) to include when saving val images


def build_geometry(cfg: TrainConfig) -> LinearProbeGeom:
    """Utility to instantiate the acquisition geometry from the config."""
    return LinearProbeGeom(
        n_det=cfg.n_det, pitch_m=cfg.pitch_m,
        t0_s=cfg.t0_s, dt_s=cfg.dt_s, n_t=cfg.n_t, c_m_s=cfg.c_m_s,
        x0_m=cfg.x0_m, y0_m=cfg.y0_m, dx_m=cfg.dx_m, dy_m=cfg.dy_m,
        nx=cfg.nx, ny=cfg.ny, array_x0_m=cfg.array_x0_m, array_y_m=cfg.array_y_m
    )


def build_projection_operators(
    cfg: TrainConfig,
    device: torch.device,
    trainable_apodization: bool = False,
) -> Tuple[DelayAndSumLinear, ForwardProjectionLinear]:
    """Construct Delay-and-Sum and forward-projection operators sharing the same LUT."""
    geom = build_geometry(cfg)
    lut = build_delay_and_sum_lut(geom, device=device)
    das = DelayAndSumLinear(geom, lut, trainable_apodization=trainable_apodization)
    fp = ForwardProjectionLinear(das)
    return das, fp


def create_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    """Build DelayAndSumTransformer (DAS + ViT) according to the provided configuration."""
    das, fp = build_projection_operators(
        cfg,
        device,
        trainable_apodization=cfg.trainable_apodization,
    )
    vit_stride = cfg.vit_stride if cfg.vit_stride is not None else cfg.vit_patch
    vit = ViTRefiner(
        in_ch=1,
        embed_dim=256,
        patch=cfg.vit_patch,
        stride=vit_stride,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        p_drop=0.1,
    )
    Hp, Wp = vit.embed.compute_grid_size(cfg.ny, cfg.nx)
    ph, pw = vit.patch_size
    sh, sw = vit.stride_size
    if Hp <= 0 or Wp <= 0:
        raise ValueError(
            "Image dimensions must be compatible with ViT patch and stride. "
            f"Got (ny={cfg.ny}, nx={cfg.nx}) with patch={vit.patch_size} and stride={vit.stride_size}."
        )
    if (cfg.ny - ph) % sh != 0 or (cfg.nx - pw) % sw != 0:
        raise ValueError(
            "Image dimensions must align with patch/stride grid. "
            f"Got (ny={cfg.ny}, nx={cfg.nx}), patch={vit.patch_size}, stride={vit.stride_size}."
        )
    vit._build_pos_embed(cfg.ny, cfg.nx, vit.embed_dim, device, dtype=torch.float32)
    freeze_das = not cfg.trainable_apodization
    variant = cfg.model_variant.lower()
    if variant == "unrolled":
        if cfg.unroll_steps < 1:
            raise ValueError("TrainConfig.unroll_steps must be >= 1 for the unrolled variant")
        model = UnrolledDelayAndSumTransformer(
            das,
            fp,
            vit,
            num_steps=cfg.unroll_steps,
            data_consistency_weight=cfg.data_consistency_weight,
            learnable_data_consistency_weight=cfg.learnable_data_consistency_weight,
            freeze_das=freeze_das,
        )
    elif variant in {"baseline", "das_transformer", "bp_transformer"}:
        if variant == "bp_transformer":
            warnings.warn(
                "TrainConfig.model_variant='bp_transformer' is deprecated; use 'das_transformer' instead.",
                DeprecationWarning,
            )
        model = DelayAndSumTransformer(das, vit, freeze_das=freeze_das)
    else:
        raise ValueError(f"Unknown model variant '{cfg.model_variant}'.")
    model = model.to(device)
    return model


def load_checkpoint(
    model: nn.Module,
    cfg: TrainConfig,
    checkpoint: str = "best.pt",
    map_location: Optional[torch.device] = None,
) -> dict:
    """Load model weights from ``cfg.work_dir`` and return the checkpoint payload."""
    ckpt_path = os.path.join(cfg.work_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if map_location is None:
        map_location = next(model.parameters()).device

    payload = torch.load(ckpt_path, map_location=map_location)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    upgraded_state, converted = adapt_vitrefiner_state_dict(state_dict)
    if converted:
        missing, unexpected = model.load_state_dict(upgraded_state, strict=False)
        if unexpected:
            warnings.warn(
                f"Ignoring unexpected keys when loading checkpoint '{checkpoint}': {unexpected}",
                RuntimeWarning,
            )
        if missing:
            warnings.warn(
                f"Missing keys when loading checkpoint '{checkpoint}': {missing}",
                RuntimeWarning,
            )
    else:
        model.load_state_dict(upgraded_state)
    model.eval()
    return payload

def main():
    seed_everything(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    # Prepare folders
    if os.path.exists(cfg.work_dir):
        shutil.rmtree(cfg.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    img_dir = os.path.join(cfg.work_dir, "val_images")
    os.makedirs(img_dir, exist_ok=True)

    # Build model (DAS module + ViT)
    model = create_model(cfg, device)

    # Optimizer (train all parameters requiring gradients)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model configuration.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Datasets / Loaders (replace with your real datasets)
    input_dir = cfg.data_root + "/" + cfg.sino_dir
    target_dir = cfg.data_root + "/" + cfg.recs_dir

    train_ds = HDF5Dataset(input_dir, target_dir, cfg.sino_min, cfg.sino_max, cfg.img_min, cfg.img_max, split="train", wavelength=cfg.wavelength, target_shape=(cfg.n_det, cfg.n_t))
    val_ds   = HDF5Dataset(input_dir, target_dir, cfg.sino_min, cfg.sino_max, cfg.img_min, cfg.img_max, split="val", wavelength=cfg.wavelength, target_shape=(cfg.n_det, cfg.n_t))

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # Log config
    with open(os.path.join(cfg.work_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Training loop (STRICT FP32: no autocast)
    best_psnr = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model,
            train_ld,
            optimizer,
            device,
            clip_grad=cfg.clip_grad,
            weight_alpha=cfg.weight_alpha,
            weight_threshold=cfg.weight_threshold,
            use_tqdm=cfg.use_tqdm,
        )
        val = validate(
            model,
            val_ld,
            device,
            epoch,
            save_dir=img_dir if cfg.save_val_images else None,
            max_save=cfg.max_val_images,
            intermediate_indices=cfg.val_intermediate_indices,
            weight_alpha=cfg.weight_alpha,
            weight_threshold=cfg.weight_threshold,
            ssim_mask_threshold=cfg.ssim_mask_threshold,
            ssim_mask_dilation=cfg.ssim_mask_dilation,
            use_tqdm=cfg.use_tqdm,
        )

        scheduler.step()

        log = {
            "epoch": epoch,
            "train_psnr": tr["psnr"], "train_ssim": tr["ssim"], "train_l1": tr["l1"],
            "train_weighted_l1": tr["weighted_l1"],
            "val_psnr": val["psnr"],   "val_ssim":  val["ssim"],  "val_l1":  val["l1"],
            "val_weighted_l1": val["weighted_l1"],
            "val_masked_ssim": val.get("masked_ssim"),
            "lr": scheduler.get_last_lr()[0]
        }
        print(json.dumps(log, indent=2))

        # Save best
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_psnr": best_psnr,
                "config": cfg.__dict__
            }, os.path.join(cfg.work_dir, "best.pt"))

    # Save last
    torch.save({
        "model": model.state_dict(),
        "epoch": cfg.epochs,
        "val_psnr": best_psnr,
        "config": cfg.__dict__
    }, os.path.join(cfg.work_dir, "last.pt"))

if __name__ == "__main__":
    main()
