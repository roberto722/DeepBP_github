"""Geometry definitions for linear probe acquisitions."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LinearProbeGeom:
    """Geometry parameters for a linear transducer array."""

    n_det: int
    pitch_m: float
    t0_s: float
    dt_s: float
    n_t: int
    c_m_s: float
    x0_m: float
    y0_m: float
    dx_m: float
    dy_m: float
    nx: int
    ny: int
    array_x0_m: float = 0.0
    array_y_m: float = 0.0

    @property
    def det_x(self) -> np.ndarray:
        """Detector x positions along the array (1D)."""

        xs = self.array_x0_m + np.arange(self.n_det) * self.pitch_m
        return xs

    @property
    def det_y(self) -> np.ndarray:
        """Detector y positions (all on same y for linear array)."""

        return np.full((self.n_det,), self.array_y_m, dtype=np.float32)


def build_delay_and_sum_lut(geom: LinearProbeGeom, device: torch.device) -> torch.Tensor:
    """Pre-compute interpolation LUT for the Delay-and-Sum beamformer."""

    xs = geom.x0_m + np.arange(geom.nx, dtype=np.float32) * geom.dx_m
    ys = geom.y0_m + np.arange(geom.ny, dtype=np.float32) * geom.dy_m
    X, Y = np.meshgrid(xs, ys)

    det_x = geom.det_x.astype(np.float32)
    det_y = geom.det_y.astype(np.float32)

    Xe = X[..., None]
    Ye = Y[..., None]
    Dx = det_x[None, None, :]
    Dy = det_y[None, None, :]

    R = np.sqrt((Xe - Dx) ** 2 + (Ye - Dy) ** 2)
    T = R / geom.c_m_s

    K = (T - geom.t0_s) / geom.dt_s
    K_floor = np.floor(K)
    Alpha = (K - K_floor).astype(np.float32)
    K_floor = K_floor.astype(np.float32)

    lut = np.stack([K_floor, Alpha], axis=-1).astype(np.float32)

    lut_t = torch.from_numpy(lut).to(device=device, dtype=torch.float32)
    return lut_t
