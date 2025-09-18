# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
# Back-Projection LUT builder
# ================================

def build_backproj_lut(geom: LinearProbeGeom, device: torch.device) -> torch.Tensor:
    """
    Precompute the (ny, nx, n_det, 2) LUT of (t_idx_floor, alpha) for linear interpolation over time.
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
    K_floor = np.floor(K).astype(np.int32)
    Alpha = (K - K_floor).astype(np.float32)

    # Pack LUT (t_idx_floor, alpha)
    lut = np.stack([K_floor, Alpha], axis=-1)       # (ny, nx, n_det, 2)

    # To torch
    lut_t = torch.from_numpy(lut).to(device)
    return lut_t


class BackProjectionLinear(nn.Module):
    """
    Differentiable DAS Back-Projection using a precomputed LUT for a linear array.
    Input:  sinogram S with shape [B, 1, n_det, n_t]
    Output: image I with shape [B, 1, ny, nx]
    Notes:
      * We linearly interpolate along the temporal dimension using (floor, alpha).
      * Out-of-range temporal indices are masked to zero contribution.
      * Summation over detectors approximates DAS (unweighted). You can add apodization later.
    """
    def __init__(self, geom: LinearProbeGeom, lut: torch.Tensor):
        super().__init__()
        self.geom = geom
        self.register_buffer("lut", lut)  # (ny, nx, n_det, 2)
        # Optional apodization per-detector (Hanning). Shape (n_det,)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(geom.n_det) / max(geom.n_det - 1, 1))
        self.register_buffer("apod", torch.from_numpy(win.astype(np.float32)))  # (n_det,)

    def forward(self, sino: torch.Tensor) -> torch.Tensor:
        """
        sino: [B, 1, n_det, n_t]
        returns: [B, 1, ny, nx]
        """
        B, C, n_det, n_t = sino.shape
        assert C == 1, "Expect sinogram with a single channel."
        assert n_det == self.geom.n_det and n_t == self.geom.n_t, "Sinogram dims mismatch geometry."

        # Extract LUT components
        k_floor = self.lut[..., 0].long()      # (ny, nx, n_det)
        alpha   = self.lut[..., 1]             # (ny, nx, n_det)

        # Clamp indices and build valid mask (0 <= k_floor < n_t-1)
        valid = (k_floor >= 0) & (k_floor < (n_t - 1))
        k0 = k_floor.clamp(0, n_t - 2)
        k1 = k0 + 1

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
        apod = self.apod  # (n_det,)
        for d in range(self.geom.n_det):
            # time indices for this detector across the image grid
            k0_d = k0e[..., d]  # (B, ny, nx)
            k1_d = k1e[..., d]
            a_d  = a[..., d]
            v_d  = ve[..., d]

            # Gather S[:, d, k] -> (B, ny, nx)
            s0 = S[:, d, :].gather(dim=1, index=k0_d.view(B, -1)).view(B, self.geom.ny, self.geom.nx)
            s1 = S[:, d, :].gather(dim=1, index=k1_d.view(B, -1)).view(B, self.geom.ny, self.geom.nx)

            # Linear interpolation
            sk = (1.0 - a_d) * s0 + a_d * s1

            # Mask out-of-range, apply apodization, and accumulate
            sk = torch.where(v_d, sk, torch.zeros_like(sk))
            out += apod[d] * sk

        # Normalize by number of detectors (and apodization sum) to keep scale stable
        norm = self.apod.sum().clamp(min=1e-6)
        out = out / norm

        return out.unsqueeze(1)  # [B, 1, ny, nx]

# ================================
# Lightweight ViT-like Refiner
# ================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""
    def __init__(self, in_ch=1, embed_dim=256, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)                          # [B, D, H/ps, W/ps]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, D]
        return x, (Hp, Wp)

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

class ViTRefiner(nn.Module):
    """
    Simple ViT-like refiner that denoises/refines the BP image.
    Input:  [B, 1, H, W]
    Output: [B, 1, H, W]
    """
    def __init__(self, in_ch=1, embed_dim=256, patch=16, depth=6, heads=8, mlp_ratio=4.0, p_drop=0.1):
        super().__init__()
        self.patch = patch
        self.embed = PatchEmbed(in_ch, embed_dim, patch)
        self.embed_dim = embed_dim
        self.pos_embed = None  # created on first forward based on (Hp, Wp)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio, p_drop)
        self.proj_out = nn.Linear(embed_dim, patch*patch)  # predict patch pixels
        self.out_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # small conv polish

    def _build_pos_embed(self, Hp, Wp, dim, device):
        pe = torch.zeros(1, Hp*Wp, dim, device=device)
        # Simple learned positional embedding
        pe = nn.Parameter(pe)
        nn.init.trunc_normal_(pe, std=0.02)
        self.pos_embed = pe

    def forward(self, x):
        # x: [B, 1, H, W]
        z, (Hp, Wp) = self.embed(x)               # z: [B, N, D]
        B, N, D = z.shape
        if self.pos_embed is None or self.pos_embed.shape[1] != N or self.pos_embed.shape[2] != D:
            self._build_pos_embed(Hp, Wp, D, x.device)
        z = z + self.pos_embed
        z = self.encoder(z)                        # [B, N, D]
        # Project tokens back to pixel patches
        pixels = self.proj_out(z)                  # [B, N, ps*ps]
        ps = self.patch
        pixels = pixels.view(B, Hp, Wp, ps, ps)    # [B, Hp, Wp, ps, ps]
        # Keep each grid dimension adjacent to its intra-patch pixels (Hp with ps, Wp with ps)
        # so that reshaping back to the image plane preserves spatial locality.
        pixels = pixels.permute(0, 1, 3, 2, 4)    # [B, Hp, ps, Wp, ps]
        img = pixels.contiguous().view(B, 1, Hp*ps, Wp*ps)
        # Final polishing conv
        img = self.out_conv(img)
        # Residual with input to stabilize training
        if img.shape == x.shape:
            img = img + x
        return img

# ================================
# Full Model: BP (fixed) + Transformer (trainable)
# ================================

class BPTransformer(nn.Module):
    def __init__(self, bp: BackProjectionLinear, vit: ViTRefiner):
        super().__init__()
        self.bp = bp
        self.vit = vit
        # Freeze BP if you want it strictly non-trainable
        for p in self.bp.parameters():
            p.requires_grad = False

    def forward(self, sino: torch.Tensor):
        bp_img = self.bp(sino)     # [B, 1, H, W]
        out = self.vit(bp_img)     # [B, 1, H, W]
        return out, bp_img


def run_inference_steps(
    model: BPTransformer,
    sinogram: torch.Tensor,
    cfg: "TrainConfig",
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized sinogram, BP image and ViT output for a given input."""
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
        pred, bp_img = model(sino_norm)

    return (
        sino_norm.detach().cpu(),
        bp_img.detach().cpu(),
        pred.detach().cpu(),
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

def save_side_by_side(pred: torch.Tensor, gt: torch.Tensor, bp: torch.Tensor,
                      out_path: str, vmin: Optional[float]=None, vmax: Optional[float]=None):
    """
    Save a side-by-side image [BP | Pred | GT] for quick inspection.
    Expects tensors in shape [1, H, W] and roughly [0,1] range.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    def to_uint8(img):
        x = img.detach().cpu().clamp(0, 1)
        if vmin is not None or vmax is not None:
            lo = 0.0 if vmin is None else vmin
            hi = 1.0 if vmax is None else vmax
            x = (x - lo) / max(hi - lo, 1e-6)
            x = x.clamp(0, 1)
        x = (x * 255.0).round().byte().squeeze(0).numpy()  # [H,W]
        return Image.fromarray(x, mode="L")

    bp_img   = to_uint8(bp[0])
    pred_img = to_uint8(pred[0])
    gt_img   = to_uint8(gt[0])

    W = bp_img.width + pred_img.width + gt_img.width
    H = max(bp_img.height, pred_img.height, gt_img.height)
    canvas = Image.new("L", (W, H))
    xoff = 0
    for im in (bp_img, pred_img, gt_img):
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


def validate(model: BPTransformer, loader: DataLoader, device: torch.device, epoch,
             save_dir: Optional[str] = None, max_save: int = 8,
             weight_alpha: float = 0.0, weight_threshold: Optional[float] = None,
             ssim_mask_threshold: Optional[float] = None, ssim_mask_dilation: int = 0):
    model.eval()
    agg = {'psnr': 0.0, 'ssim': 0.0, 'l1': 0.0, 'weighted_l1': 0.0}
    compute_masked_ssim = ssim_mask_threshold is not None
    if compute_masked_ssim:
        agg['masked_ssim'] = 0.0
    n = 0
    saved = 0
    with torch.no_grad():  # NOTE: no autocast anywhere (pure FP32)
        for i, (sino, img) in enumerate(loader):
            sino = sino.to(device, non_blocking=True)
            img  = img.to(device, non_blocking=True)

            pred, bp = model(sino)            # forward in FP32
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

            # Save a few side-by-side for inspection
            if save_dir is not None and saved < max_save:
                for b in range(min(bs, max_save - saved)):
                    out_path = os.path.join(save_dir, f"val_epoch_{epoch}_{i:04d}_più corta{b:02d}.png")
                    save_side_by_side(pred[b], img[b], bp[b], out_path)
                    saved += 1

    for k in agg:
        agg[k] /= max(n, 1)
    return agg


def train_one_epoch(model: BPTransformer, loader: DataLoader, optimizer, device: torch.device,
                    clip_grad: float = 1.0, weight_alpha: float = 0.0,
                    weight_threshold: Optional[float] = None):
    model.train()
    agg = {'psnr': 0.0, 'ssim': 0.0, 'l1': 0.0, 'weighted_l1': 0.0}
    n = 0
    for i, (sino, img) in enumerate(loader):
        sino = sino.to(device, non_blocking=True)
        img  = img.to(device, non_blocking=True)

        pred, _ = model(sino)                     # forward in FP32
        weights = compute_intensity_weights(img, weight_alpha, weight_threshold)
        loss_weighted_l1 = torch.mean(weights * torch.abs(pred - img))

        optimizer.zero_grad(set_to_none=True)
        loss_weighted_l1.backward()               # backprop in FP32
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.vit.parameters(), clip_grad)
        optimizer.step()

        # metrics on-the-fly
        with torch.no_grad():
            bs = sino.size(0)
            n += bs
            agg['weighted_l1'] += loss_weighted_l1.item() * bs
            agg['l1']   += F.l1_loss(pred, img).item() * bs
            agg['psnr'] += psnr(pred, img).mean().item() * bs
            agg['ssim'] += ssim(pred, img).mean().item() * bs

    for k in agg:
        agg[k] /= max(n, 1)
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
    dx_m: float = 0.00015               # Spaziatura tra pixel sull’asse x.
    dy_m: float = 0.00015               # Spaziatura tra pixel sull’asse y.
    nx: int = 256
    ny: int = 256
    array_x0_m: float = -0.019         # align first element with left image border (example)
    array_y_m: float = 0.0
    wavelength: int = 800

    # Training
    epochs: int = 50
    batch_size: int = 4
    lr: float = 2e-4
    num_workers: int = 4
    clip_grad: float = 1.0
    weight_alpha: float = 1.0
    weight_threshold: Optional[float] = 0.5
    ssim_mask_threshold: Optional[float] = 0.5
    ssim_mask_dilation: int = 0

    # --- Global scaling (per-domain) ---
    # If any of these is None, stats will be computed from the training set.
    sino_min: float = -11.0322
    sino_max: float = 12.5394
    img_min: float = 0.0
    img_max: float = 316.9658

    # Paths
    work_dir: str = "./runs/bp_transformer_fp32"
    data_root: str = "E:/Scardigno/datasets_transformer_proj"
    sino_dir: str = "Forearm2000_hdf5/train_val_tst"
    recs_dir: str = "Forearm2000_recs/L1_Shearlet"
    save_val_images: bool = True
    max_val_images: int = 2


def build_geometry(cfg: TrainConfig) -> LinearProbeGeom:
    """Utility to instantiate the acquisition geometry from the config."""
    return LinearProbeGeom(
        n_det=cfg.n_det, pitch_m=cfg.pitch_m,
        t0_s=cfg.t0_s, dt_s=cfg.dt_s, n_t=cfg.n_t, c_m_s=cfg.c_m_s,
        x0_m=cfg.x0_m, y0_m=cfg.y0_m, dx_m=cfg.dx_m, dy_m=cfg.dy_m,
        nx=cfg.nx, ny=cfg.ny, array_x0_m=cfg.array_x0_m, array_y_m=cfg.array_y_m
    )


def create_model(cfg: TrainConfig, device: torch.device) -> BPTransformer:
    """Build BPTransformer (BP + ViT) according to the provided configuration."""
    geom = build_geometry(cfg)
    lut = build_backproj_lut(geom, device=device)
    bp = BackProjectionLinear(geom, lut)
    vit = ViTRefiner(in_ch=1, embed_dim=256, patch=16, depth=6, heads=8, mlp_ratio=4.0, p_drop=0.1)
    if cfg.ny % vit.patch != 0 or cfg.nx % vit.patch != 0:
        raise ValueError(
            f"Image dimensions (ny={cfg.ny}, nx={cfg.nx}) must be divisible by ViT patch size {vit.patch}."
        )
    vit._build_pos_embed(cfg.ny // vit.patch, cfg.nx // vit.patch, vit.embed_dim, device)
    model = BPTransformer(bp, vit).to(device)
    return model


def load_checkpoint(
    model: BPTransformer,
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
    model.load_state_dict(state_dict)
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

    # Build model (BP module + ViT)
    model = create_model(cfg, device)

    # Optimizer (only ViT is trainable)
    optimizer = torch.optim.AdamW(model.vit.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-3)
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
        )
        val = validate(model, val_ld, device, epoch,
                       save_dir=img_dir if cfg.save_val_images else None,
                       max_save=cfg.max_val_images,
                       weight_alpha=cfg.weight_alpha,
                       weight_threshold=cfg.weight_threshold,
                       ssim_mask_threshold=cfg.ssim_mask_threshold,
                       ssim_mask_dilation=cfg.ssim_mask_dilation)

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
