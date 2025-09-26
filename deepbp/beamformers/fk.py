"""Frequency-wavenumber migration operators."""
import math
import warnings
from typing import List, Optional, Tuple

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
        static_output_scale: Optional[float] = None,
        static_output_shift: Optional[float] = None,
        output_norm_scale_init: Optional[float] = None,
        output_norm_shift_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.geom = geom
        self.trainable_apodization = trainable_apodization
        self.per_channel_apodization = per_channel_apodization
        self.learnable_output_normalization = learnable_output_normalization
        self.static_output_scale = static_output_scale
        self.static_output_shift = static_output_shift

        if self.learnable_output_normalization and (
            self.static_output_scale is not None or self.static_output_shift is not None
        ):
            raise ValueError(
                "Static output normalization cannot be combined with learnable output normalization."
            )

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

        x_phase_coords = xs - geom.array_x0_m
        y_phase_coords = ys - geom.array_y_m
        self.register_buffer("x_phase_coords", x_phase_coords)
        self.register_buffer("y_phase_coords", y_phase_coords)

        omega_term = (omega / self.geom.c_m_s) ** 2
        kx_sq = kx ** 2
        kz_sq = omega_term.view(1, -1) - kx_sq.view(-1, 1)
        prop_mask = (kz_sq > 0).to(dtype=torch.float32)
        kz_sq = torch.clamp(kz_sq, min=0.0)
        kz = torch.sqrt(kz_sq)
        phase_x = torch.exp(1j * (kx.view(-1, 1) * x_phase_coords.view(1, -1)))
        phase_y = torch.exp(1j * (y_phase_coords.view(-1, 1, 1) * kz.view(1, geom.n_det, -1)))

        self.register_buffer("prop_mask", prop_mask)
        self.register_buffer("kz", kz)
        self.register_buffer("phase_x", phase_x.to(dtype=torch.complex64))
        self.register_buffer("phase_y", phase_y.to(dtype=torch.complex64))

        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(geom.n_det) / max(geom.n_det - 1, 1))
        apod = torch.from_numpy(win.astype(np.float32))
        if per_channel_apodization:
            apod = apod.unsqueeze(0)
        if trainable_apodization:
            self.apod = nn.Parameter(apod)
        else:
            self.register_buffer("apod", apod)

        if self.learnable_output_normalization:
            scale_value = 1.0
            shift_value = 0.0

            if output_norm_scale_init is not None:
                if output_norm_scale_init <= 0:
                    raise ValueError("output_norm_scale_init must be positive for softplus inversion")
                scale_tensor = torch.tensor(output_norm_scale_init, dtype=torch.float32)
                scale_value = torch.log(torch.expm1(scale_tensor))
            if output_norm_shift_init is not None:
                shift_value = float(output_norm_shift_init)

            self.output_scale = nn.Parameter(torch.tensor(scale_value, dtype=torch.float32))
            self.output_shift = nn.Parameter(torch.tensor(shift_value, dtype=torch.float32))

        self._cached_norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

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

    def get_phase_matrices(
        self, device: torch.device, dtype: torch.dtype
    ) -> List[torch.Tensor]:
        """Return cached phase matrices converted to the requested dtype/device."""

        phase_x = self.phase_x.to(device=device, dtype=dtype)
        phase_y = self.phase_y.to(device=device, dtype=dtype)
        return [phase_x, phase_y]

    def _normalize_sinogram(
        self,
        sino: torch.Tensor,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_cache: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize a sinogram using per-sample statistics.

        Parameters
        ----------
        sino:
            Input sinogram of shape ``[B, C, n_det, n_t]``.
        stats:
            Optional tuple ``(mean, std)`` computed previously. When provided the
            sinogram is normalized using these cached statistics.
        update_cache:
            If ``True``, cache the computed statistics for later reuse.

        Returns
        -------
        normalized, (mean, std)
            The normalized sinogram along with the statistics that were used.
        """

        if stats is not None:
            mean, std = stats
            centered = sino - mean
        else:
            eps = torch.finfo(sino.dtype).eps
            mean = sino.mean(dim=(-1, -2), keepdim=True)
            centered = sino - mean
            var = centered.pow(2).mean(dim=(-1, -2), keepdim=True)
            std = torch.sqrt(var + eps)

        normalized = centered / std

        if stats is None:
            stats = (mean, std)

        if update_cache:
            self._cached_norm_stats = stats

        return normalized, stats

    def normalize_with_cached_stats(self, sino: torch.Tensor) -> torch.Tensor:
        """Normalize ``sino`` using the cached statistics without mutating them."""

        if self._cached_norm_stats is None:
            raise RuntimeError(
                "Normalization statistics are unavailable; call the beamformer "
                "on a measured sinogram before requesting cached normalization."
            )

        normalized, _ = self._normalize_sinogram(
            sino, stats=self._cached_norm_stats, update_cache=False
        )
        return normalized

    def forward(
        self,
        sino: torch.Tensor,
        *,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_cache: bool = True,
        pre_normalized: bool = False,
    ) -> torch.Tensor:
        """Apply f-k migration to a sinogram of shape ``[B, C, n_det, n_t]``.

        Parameters
        ----------
        sino:
            Input sinogram. If ``pre_normalized`` is ``False`` the sinogram will
            be normalized internally using either the provided ``stats`` or
            freshly computed statistics.
        stats:
            Optional cached normalization statistics to reuse. Ignored when the
            input is already normalized.
        update_cache:
            When ``True`` (default) the internally computed statistics are
            cached for later reuse. This must be ``False`` when ``stats`` are
            supplied or when ``pre_normalized`` is ``True``.
        pre_normalized:
            If ``True`` the sinogram is assumed to already be normalized and the
            cached statistics are required to be present.
        """

        B, C, n_det, n_t = sino.shape
        assert n_det == self.geom.n_det and n_t == self.geom.n_t, "Sinogram dims mismatch geometry."

        dtype = sino.dtype
        device = sino.device
        eps = torch.finfo(dtype).eps

        if pre_normalized:
            if self._cached_norm_stats is None:
                raise RuntimeError(
                    "Cached normalization statistics are required for pre-normalized inputs."
                )
            if update_cache:
                raise ValueError("Cannot update cached stats when input is pre-normalized.")
            sino_norm = sino
        else:
            if stats is None and not update_cache:
                stats = self._cached_norm_stats
            sino_norm, stats = self._normalize_sinogram(
                sino,
                stats=stats,
                update_cache=update_cache,
            )

        time_window = self.time_window.to(device=device, dtype=dtype)
        sino_windowed = sino_norm * time_window.view(1, 1, 1, -1)

        apod = self.get_apodization(dtype, device, C)
        apod_view = apod.unsqueeze(0).unsqueeze(-1)
        sino_weighted = sino_windowed * apod_view

        spec_t = torch.fft.rfft(sino_weighted, n=self.n_fft, dim=-1)
        spec_fk = torch.fft.fft(spec_t, dim=2)

        real_dtype = spec_fk.real.dtype
        freq_weight = self.freq_weight.to(device=device, dtype=real_dtype)
        freq_weight_sum = torch.clamp(self.freq_weight_sum.to(device=device, dtype=dtype), min=eps)
        freq_weight = freq_weight.view(1, 1, 1, -1).to(dtype=spec_fk.dtype)

        prop_mask = self.prop_mask.to(device=device, dtype=spec_fk.dtype)
        spec_fk = spec_fk * prop_mask.view(1, 1, self.geom.n_det, -1)

        phase_x, phase_y = self.get_phase_matrices(device, spec_fk.dtype)
        weighted_spec = spec_fk * freq_weight
        band = torch.einsum("bcdw,ydw->bcyd", weighted_spec, phase_y)
        band = band.reshape(B * C * self.geom.ny, self.geom.n_det)
        img_complex = torch.matmul(band, phase_x)
        img_complex = img_complex.view(B, C, self.geom.ny, self.geom.nx)

        img_mag = img_complex.abs().to(dtype)

        apod_sum = torch.clamp(apod.sum(dim=-1, keepdim=True), min=eps)
        norm = apod_sum.view(1, C, 1, 1) * freq_weight_sum.reshape(1, 1, 1, 1)
        img_mag = img_mag / norm

        # print(f"Min: {img_mag.min()}, Max: {img_mag.max()}")

        if self.learnable_output_normalization:
            scale = F.softplus(self.output_scale).to(dtype=img_mag.dtype, device=img_mag.device)
            shift = self.output_shift.to(dtype=img_mag.dtype, device=img_mag.device)
            img_mag = torch.sigmoid(scale * (img_mag - shift))
        else:
            if self.static_output_shift is not None:
                shift = torch.as_tensor(
                    self.static_output_shift, dtype=img_mag.dtype, device=img_mag.device
                )
                img_mag = img_mag - shift
            if self.static_output_scale is not None:
                scale = torch.as_tensor(
                    self.static_output_scale, dtype=img_mag.dtype, device=img_mag.device
                )
                if torch.any(scale == 0):
                    raise ValueError("static_output_scale must be non-zero for normalization")
                img_mag = img_mag / scale

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

        phase_x_cache, phase_y_cache = self.migration.get_phase_matrices(device, complex_dtype)

        n_kx = phase_x_cache.shape[0]
        n_freq = phase_y_cache.shape[-1]

        img_line = img[:, 0, :, :].to(dtype=working_dtype)
        phase_x = torch.conj(phase_x_cache).transpose(0, 1)
        img_line_flat = img_line.reshape(B * ny, nx)
        img_fft = torch.matmul(img_line_flat.to(dtype=complex_dtype), phase_x)
        img_fft = img_fft.view(B, ny, n_kx)
        dx = torch.as_tensor(self.geom.dx_m, device=device, dtype=working_dtype)
        img_fft = img_fft.to(dtype=complex_dtype) * dx.to(dtype=complex_dtype)

        prop_mask = self.migration.prop_mask.to(device=device, dtype=complex_dtype)
        phase_y = torch.conj(phase_y_cache)
        phase_y = phase_y * prop_mask.view(1, n_kx, n_freq)

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

        cached_stats = getattr(self.migration, "_cached_norm_stats", None)
        sino_norm, _ = self.migration._normalize_sinogram(
            sino,
            stats=cached_stats,
            update_cache=False,
        )

        return sino_norm.to(dtype=dtype_in)
