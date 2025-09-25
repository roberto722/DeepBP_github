"""Frequency-wavenumber migration operators."""
import math
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import LinearProbeGeom


class FkMigrationLinear(nn.Module):
    """Frequency-wavenumber migration for linear probe acquisitions."""

    def __init__(
        self,
        geom: LinearProbeGeom,
        trainable_apodization: bool = False,
        per_channel_apodization: bool = False,
        fft_pad: int = 0,
        window: Optional[str] = None,
        learnable_output_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.geom = geom
        self.trainable_apodization = trainable_apodization
        self.per_channel_apodization = per_channel_apodization
        self.learnable_output_normalization = learnable_output_normalization

        if fft_pad is None:
            fft_pad = 0
        if fft_pad < 0:
            raise ValueError("fft_pad must be >= 0 for FkMigrationLinear")
        self.fft_pad = int(fft_pad)
        self.n_fft = geom.n_t + self.fft_pad
        if self.n_fft <= 0:
            raise ValueError("Invalid FFT size computed for FkMigrationLinear")

        window_tensor = self._build_time_window(window, geom.n_t)
        self.register_buffer("time_window", window_tensor)
        self.window_type = None if window is None else window.lower()

        freq = torch.fft.rfftfreq(self.n_fft, d=geom.dt_s)
        omega = (2.0 * math.pi * freq).to(dtype=torch.float32)
        self.register_buffer("omega", omega)

        weight = torch.ones_like(omega)
        if omega.numel() > 2:
            weight[1:-1] = 2.0
        if self.n_fft % 2 == 0 and omega.numel() > 1:
            weight[-1] = 1.0
        self.register_buffer("freq_weight", weight)
        self.register_buffer("freq_weight_sum", weight.sum())

        kx = torch.fft.fftfreq(geom.n_det, d=geom.pitch_m)
        kx = (2.0 * math.pi * kx).to(dtype=torch.float32)
        self.register_buffer("kx", kx)

        xs = geom.x0_m + torch.arange(geom.nx, dtype=torch.float32) * geom.dx_m
        ys = geom.y0_m + torch.arange(geom.ny, dtype=torch.float32) * geom.dy_m
        self.register_buffer("x_coords", xs)
        self.register_buffer("y_coords", ys)

        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(geom.n_det) / max(geom.n_det - 1, 1))
        apod = torch.from_numpy(win.astype(np.float32))
        if per_channel_apodization:
            apod = apod.unsqueeze(0)
        if trainable_apodization:
            self.apod = nn.Parameter(apod)
        else:
            self.register_buffer("apod", apod)

        if self.learnable_output_normalization:
            self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.output_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _build_time_window(window: Optional[str], length: int) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Window length must be positive for FkMigrationLinear")
        if window is None:
            values = torch.ones(length, dtype=torch.float32)
        else:
            name = window.lower()
            if name in {"hann", "hanning"}:
                values = torch.hann_window(length, periodic=False, dtype=torch.float32)
            elif name == "hamming":
                values = torch.hamming_window(length, periodic=False, dtype=torch.float32)
            elif name == "blackman":
                values = torch.blackman_window(length, periodic=False, dtype=torch.float32)
            elif name in {"rect", "rectangular", "none"}:
                values = torch.ones(length, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported window '{window}' for FkMigrationLinear")
        return values.to(dtype=torch.float32)

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

        mean = sino.mean(dim=(-1, -2), keepdim=True)
        sino_norm = sino - mean
        var = sino_norm.pow(2).mean(dim=(-1, -2), keepdim=True)
        std = torch.sqrt(var + eps)
        sino_norm = sino_norm / std

        time_window = self.time_window.to(device=device, dtype=dtype)
        sino_windowed = sino_norm * time_window.view(1, 1, 1, -1)

        apod = self.get_apodization(dtype, device, C)
        apod_view = apod.unsqueeze(0).unsqueeze(-1)
        sino_weighted = sino_windowed * apod_view

        spec_t = torch.fft.rfft(sino_weighted, n=self.n_fft, dim=-1)
        spec_fk = torch.fft.fft(spec_t, dim=2)

        real_dtype = spec_fk.real.dtype
        omega = self.omega.to(device=device, dtype=real_dtype)
        kx = self.kx.to(device=device, dtype=real_dtype)

        omega_term = (omega / self.geom.c_m_s) ** 2
        kx_sq = kx ** 2

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
            band = weighted.sum(dim=-1)
            band = band.reshape(B * C, self.geom.n_det)
            line = torch.matmul(band, phase_x)
            line = line.view(B, C, self.geom.nx)
            lines.append(line)

        img_complex = torch.stack(lines, dim=2)
        img_mag = img_complex.abs().to(dtype)

        apod_sum = torch.clamp(apod.sum(dim=-1, keepdim=True), min=eps)
        norm = apod_sum.view(1, C, 1, 1) * freq_weight_sum.reshape(1, 1, 1, 1)
        img_mag = img_mag / norm

        # print(f"Min: {img_mag.min()}, Max: {img_mag.max()}")

        if self.learnable_output_normalization:
            scale = F.softplus(self.output_scale).to(dtype=img_mag.dtype, device=img_mag.device)
            shift = self.output_shift.to(dtype=img_mag.dtype, device=img_mag.device)
            img_mag = torch.sigmoid(scale * (img_mag - shift))

        return img_mag


class ForwardProjectionFk(nn.Module):
    """Forward projection in frequency-wavenumber (f-k) domain."""

    def __init__(self, migration: FkMigrationLinear) -> None:
        super().__init__()
        self.geom = migration.geom
        self.n_fft = getattr(migration, "n_fft", migration.geom.n_t)
        object.__setattr__(self, "_migration", migration)
        self._warned_sampling_mismatch = False

    @property
    def migration(self) -> FkMigrationLinear:
        return self._migration

    def get_apodization(self, dtype: torch.dtype, device: torch.device, channels: int) -> torch.Tensor:
        return self.migration.get_apodization(dtype, device, channels)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Project an image into the sinogram domain using the f-k model."""

        B, C, ny, nx = img.shape
        assert C == 1, "Expect image with a single channel."
        assert ny == self.geom.ny and nx == self.geom.nx, "Image dims mismatch geometry."

        if (
            (nx != self.geom.n_det)
            or not math.isclose(
                self.geom.dx_m, self.geom.pitch_m, rel_tol=1e-6, abs_tol=1e-9
            )
        ) and not self._warned_sampling_mismatch:
            warnings.warn(
                "ForwardProjectionFk received an image grid that does not match the detector "
                "sampling. The forward projection will internally resample along the lateral "
                "axis; please verify that geometry parameters are correct for best accuracy.",
                RuntimeWarning,
            )
            self._warned_sampling_mismatch = True

        dtype_in = img.dtype
        if dtype_in in (torch.float16, torch.bfloat16):
            working_dtype = torch.float32
        else:
            working_dtype = dtype_in

        if working_dtype == torch.float32:
            complex_dtype = torch.complex64
        elif working_dtype == torch.float64:
            complex_dtype = torch.complex128
        else:
            raise TypeError(f"Unsupported dtype {dtype_in} for ForwardProjectionFk")

        device = img.device

        kx = self.migration.kx.to(device=device, dtype=working_dtype)
        omega = self.migration.omega.to(device=device, dtype=working_dtype)
        x_coords = self.migration.x_coords.to(device=device, dtype=working_dtype)
        y_coords = self.migration.y_coords.to(device=device, dtype=working_dtype)

        x_phase_coords = x_coords - self.geom.array_x0_m
        y_phase_coords = y_coords - self.geom.array_y_m

        n_kx = kx.numel()
        n_freq = omega.numel()

        img_line = img[:, 0, :, :].to(dtype=working_dtype)
        phase_x = torch.exp(
            -1j * (kx.view(-1, 1) * x_phase_coords.view(1, -1))
        ).to(dtype=complex_dtype)
        img_line_flat = img_line.reshape(B * ny, nx)
        img_fft = torch.matmul(
            img_line_flat.to(dtype=complex_dtype), phase_x.transpose(0, 1)
        )
        img_fft = img_fft.view(B, ny, n_kx)
        dx = torch.as_tensor(self.geom.dx_m, device=device, dtype=working_dtype)
        img_fft = img_fft.to(dtype=complex_dtype) * dx.to(dtype=complex_dtype)

        omega_term = (omega / self.geom.c_m_s) ** 2
        kx_sq = kx ** 2
        kz_sq = omega_term.view(1, -1) - kx_sq.view(-1, 1)
        prop_mask = kz_sq > 0
        kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))
        phase = y_phase_coords.view(ny, 1, 1) * kz.view(1, n_kx, n_freq)
        phase_y = torch.exp(-1j * phase)
        phase_y = phase_y.to(dtype=complex_dtype)
        phase_y = phase_y * prop_mask.view(1, n_kx, n_freq).to(dtype=complex_dtype)

        img_fft = img_fft.unsqueeze(1)
        spec_fk = torch.einsum("bcyk,ykf->bckf", img_fft, phase_y)
        dy = torch.as_tensor(self.geom.dy_m, device=device, dtype=complex_dtype)
        spec_fk = spec_fk * dy

        freq_weight = self.migration.freq_weight.to(device=device, dtype=working_dtype)
        spec_fk = spec_fk * freq_weight.view(1, 1, 1, -1).to(dtype=complex_dtype)

        spec_t = torch.fft.ifft(spec_fk, dim=2)
        sino = torch.fft.irfft(spec_t, n=self.n_fft, dim=-1)
        if self.n_fft != self.geom.n_t:
            sino = sino[..., : self.geom.n_t]

        apod = self.get_apodization(working_dtype, device, channels=1)
        eps = torch.finfo(apod.dtype).tiny
        norm = torch.clamp(apod.sum(dim=-1, keepdim=True), min=eps)
        sino = sino / norm.view(1, 1, 1, 1)

        return sino.to(dtype=dtype_in)
