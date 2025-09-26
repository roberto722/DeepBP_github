"""Tests for shared f-k normalization utilities."""

from __future__ import annotations

from typing import Tuple
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from deepbp.beamformers.fk import FkMigrationLinear, ForwardProjectionFk
from deepbp.geometry import LinearProbeGeom


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


def _manual_stats(sino: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(sino.dtype).eps
    mean = sino.mean(dim=(-1, -2), keepdim=True)
    centered = sino - mean
    var = centered.pow(2).mean(dim=(-1, -2), keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std


def test_normalize_helper_matches_manual_statistics() -> None:
    geom = _make_geometry()
    migration = FkMigrationLinear(geom)

    sino = torch.randn(2, 3, geom.n_det, geom.n_t)

    normalized, stats = migration._normalize_sinogram(sino, update_cache=True)
    manual_mean, manual_std = _manual_stats(sino)
    manual_norm = (sino - manual_mean) / manual_std

    assert migration._cached_norm_stats is not None
    assert torch.allclose(stats[0], manual_mean)
    assert torch.allclose(stats[1], manual_std)
    assert torch.allclose(normalized, manual_norm)


def test_forward_projection_reuses_cached_normalization_stats() -> None:
    geom = _make_geometry()
    migration = FkMigrationLinear(geom)
    projector = ForwardProjectionFk(migration)

    measured = torch.randn(1, 1, geom.n_det, geom.n_t, requires_grad=True)
    # Sets the cached normalization statistics.
    migration.forward(measured)

    cached_stats = migration._cached_norm_stats
    assert cached_stats is not None

    image = torch.randn(1, 1, geom.ny, geom.nx, requires_grad=True)

    with mock.patch.object(migration, "_normalize_sinogram", wraps=migration._normalize_sinogram) as norm_mock:
        sino = projector.forward(image)

    assert norm_mock.called
    called_kwargs = norm_mock.call_args.kwargs
    assert "stats" in called_kwargs
    assert called_kwargs["stats"] is cached_stats

    # Gradients should flow back to the measured sinogram through the cached statistics.
    loss = sino.sum()
    loss.backward()

    assert measured.grad is not None
    assert torch.all(torch.isfinite(measured.grad))


def test_learnable_output_norm_initialization_from_config() -> None:
    geom = _make_geometry()
    desired_scale = 2.5
    desired_shift = -0.75

    migration = FkMigrationLinear(
        geom,
        learnable_output_normalization=True,
        output_norm_scale_init=desired_scale,
        output_norm_shift_init=desired_shift,
    )

    initialized_scale = F.softplus(migration.output_scale.detach())
    initialized_shift = migration.output_shift.detach()

    assert torch.allclose(initialized_scale, torch.tensor(desired_scale, dtype=torch.float32))
    assert torch.allclose(initialized_shift, torch.tensor(desired_shift, dtype=torch.float32))
