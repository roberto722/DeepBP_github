"""Regression tests for the unrolled transformer residual scaling."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepbp.beamformers.fk import FkMigrationLinear, ForwardProjectionFk
from deepbp.geometry import LinearProbeGeom
from deepbp.models.transformer import UnrolledDelayAndSumTransformer


def _make_geometry() -> LinearProbeGeom:
    return LinearProbeGeom(
        n_det=4,
        pitch_m=0.0003,
        t0_s=0.0,
        dt_s=1e-7,
        n_t=8,
        c_m_s=1500.0,
        x0_m=0.0,
        y0_m=0.0,
        dx_m=0.0003,
        dy_m=0.0003,
        nx=4,
        ny=4,
        array_x0_m=0.0,
        array_y_m=0.0,
    )


def _variance(x: torch.Tensor) -> torch.Tensor:
    return x.var(dim=(-1, -2, -3), unbiased=False)


def test_unrolled_residual_matches_forward_projection_variance() -> None:
    geom = _make_geometry()
    beamformer = FkMigrationLinear(geom)
    forward = ForwardProjectionFk(beamformer)

    model = UnrolledDelayAndSumTransformer(
        beamformer_module=beamformer,
        forward_module=forward,
        vit_module=torch.nn.Identity(),
        num_steps=1,
        data_consistency_weight=1.0,
        freeze_beamformer=False,
    )

    torch.manual_seed(42)
    sino = torch.randn(1, 1, geom.n_det, geom.n_t)

    _, x0, _ = model(sino)

    sino_normalized = beamformer.normalize_with_cached_stats(sino)
    sino_est = forward(x0)
    sino_residual = sino_normalized - sino_est

    var_est = _variance(sino_est)
    var_residual = _variance(sino_residual)

    assert torch.all(var_est > 0)

    ratio = var_residual / var_est

    # Residual variance should stay within a small factor of the projection variance.
    assert torch.all(ratio > 0.1)
    assert torch.all(ratio < 10.0)
