"""Differentiable delay-and-sum beamformer and its adjoint."""

import numpy as np
import torch
import torch.nn as nn

from ..geometry import LinearProbeGeom


class DelayAndSumLinear(nn.Module):
    """Differentiable Delay-and-Sum beamformer using a precomputed LUT."""

    def __init__(
        self,
        geom: LinearProbeGeom,
        lut: torch.Tensor,
        trainable_apodization: bool = False,
    ) -> None:
        super().__init__()
        self.geom = geom
        self.trainable_apodization = trainable_apodization
        lut = lut.to(dtype=torch.float32)
        self.register_buffer("lut", lut)

        k_floor = torch.floor(lut[..., 0])
        valid = (k_floor >= 0) & (k_floor < (geom.n_t - 1))
        max_idx = max(geom.n_t - 2, 0)
        k0 = torch.clamp(k_floor, 0, max_idx).to(dtype=torch.long)
        alpha = lut[..., 1].to(dtype=torch.float32)
        self.register_buffer("k0", k0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("valid", valid)

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
        """Apply the DAS beamformer to ``sino``."""

        B, C, n_det, n_t = sino.shape
        assert C == 1, "Expect sinogram with a single channel."
        assert n_det == self.geom.n_det and n_t == self.geom.n_t, "Sinogram dims mismatch geometry."

        k0 = self.k0
        k1 = k0 + 1
        alpha = self.alpha
        if alpha.dtype != sino.dtype:
            alpha = alpha.to(dtype=sino.dtype)
        valid = self.valid

        S = sino[:, 0, :, :]
        out = torch.zeros((B, self.geom.ny, self.geom.nx), device=sino.device, dtype=sino.dtype)

        k0e = k0.unsqueeze(0).expand(B, -1, -1, -1)
        k1e = k1.unsqueeze(0).expand(B, -1, -1, -1)
        a = alpha.unsqueeze(0).expand(B, -1, -1, -1)
        ve = valid.unsqueeze(0).expand(B, -1, -1, -1)

        apod = self.get_apodization(sino.dtype, sino.device)
        for d in range(self.geom.n_det):
            k0_d = k0e[..., d]
            k1_d = k1e[..., d]
            a_d = a[..., d]
            v_d = ve[..., d]

            s0 = S[:, d, :].gather(dim=1, index=k0_d.reshape(B, -1)).reshape(B, self.geom.ny, self.geom.nx)
            s1 = S[:, d, :].gather(dim=1, index=k1_d.reshape(B, -1)).reshape(B, self.geom.ny, self.geom.nx)

            sk = (1.0 - a_d) * s0 + a_d * s1
            sk = sk * v_d.to(sk.dtype)
            out += apod[d] * sk

        norm = torch.clamp(apod.sum(), min=torch.finfo(apod.dtype).tiny)
        out = out / norm

        return out.unsqueeze(1)


class ForwardProjectionLinear(nn.Module):
    """Forward projection (image -> sinogram) adjoint of the DAS beamformer."""

    def __init__(self, das: DelayAndSumLinear) -> None:
        super().__init__()
        self.das = das
        self.geom = das.geom

    def forward(self, img: torch.Tensor) -> torch.Tensor:
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

        img_flat = img[:, 0, :, :].reshape(B, -1)
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
