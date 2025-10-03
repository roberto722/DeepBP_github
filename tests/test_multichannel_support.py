"""Tests covering multi-channel beamformer outputs in inference and visualization."""

import math
from typing import Optional, Tuple

import pytest

Image = pytest.importorskip("PIL.Image")

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from deepbp.config import TrainConfig
from deepbp.inference import run_inference_steps
from deepbp.visualization import save_side_by_side


class _DummyModel(nn.Module):
    def __init__(self, output_shape: Tuple[int, int]):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self._output_shape = output_shape

    def forward(
        self, sino: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch = sino.shape[0]
        h, w = self._output_shape
        device = sino.device
        dtype = sino.dtype

        base = torch.linspace(0.0, 1.0, steps=h * w, device=device, dtype=dtype)
        base = base.view(1, 1, h, w).repeat(batch, 1, 1, 1)
        beamformer = torch.stack(
            (
                base.squeeze(1),
                torch.ones_like(base.squeeze(1)) * 0.5,
            ),
            dim=1,
        )
        pred = base * 0.25
        return pred, beamformer, None


def _normalization_range(
    channel: torch.Tensor,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[float, float]:
    x = channel.clone().to(dtype=torch.float32)
    local_min = local_max = None
    if vmin is None or vmax is None:
        finite_mask = torch.isfinite(x)
        if finite_mask.any():
            valid = x[finite_mask]
            local_min, local_max = torch.aminmax(valid)
        else:
            local_min = torch.tensor(0.0, dtype=x.dtype)
            local_max = torch.tensor(1.0, dtype=x.dtype)

    if vmin is not None:
        lo = float(vmin)
    elif local_min is not None:
        lo = float(local_min.item())
    else:
        lo = 0.0

    if vmax is not None:
        hi = float(vmax)
    elif local_max is not None:
        hi = float(local_max.item())
    else:
        hi = lo

    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi):
        hi = lo
    if hi - lo < 1e-6:
        hi = lo + 1e-6

    return lo, hi


def _expected_uint8(
    channel: torch.Tensor,
    normalization: Optional[Tuple[float, float]] = None,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> torch.Tensor:
    x = channel.clone().to(dtype=torch.float32)

    if normalization is None:
        lo, hi = _normalization_range(x, vmin=vmin, vmax=vmax)
    else:
        lo, hi = normalization
        if not math.isfinite(lo):
            lo = 0.0
        if not math.isfinite(hi):
            hi = lo
        if hi - lo < 1e-6:
            hi = lo + 1e-6

    x = torch.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
    x = (x - lo) / (hi - lo)
    x = x.clamp(0.0, 1.0)
    return (x * 255.0).round().to(dtype=torch.uint8)


def test_run_inference_steps_returns_canonical_initial_and_stack():
    model = _DummyModel(output_shape=(3, 4))
    cfg = TrainConfig()
    sino = torch.randn(1, 1, 2, 2)

    (
        _,
        initial_img,
        pred,
        iter_imgs,
        beamformer_stack,
    ) = run_inference_steps(model, sino, cfg, normalize=False)

    assert initial_img is not None
    assert beamformer_stack is not None
    assert beamformer_stack.shape[1] == 2
    assert initial_img.shape[1] == 1
    assert torch.allclose(initial_img[:, 0], beamformer_stack[:, 0])
    assert iter_imgs is not None
    assert torch.allclose(iter_imgs[0], initial_img)
    assert pred.shape[0] == sino.shape[0]


def test_save_side_by_side_multichannel_uses_first_channel(tmp_path):
    pred = torch.zeros(1, 1, 2, 2)
    gt = torch.zeros(1, 1, 2, 2)
    initial = torch.stack(
        (
            torch.tensor([[0.0, 1.0], [0.5, 0.75]], dtype=torch.float32),
            torch.full((2, 2), 0.25, dtype=torch.float32),
        ),
        dim=0,
    )
    iter_tensor = torch.stack(
        (
            torch.tensor([[1.0, 0.0], [0.25, 0.5]], dtype=torch.float32),
            torch.full((2, 2), -1.0, dtype=torch.float32),
        ),
        dim=0,
    )

    out_path = tmp_path / "viz.png"
    save_side_by_side(
        pred[0],
        gt[0],
        initial,
        str(out_path),
        iter_steps=[(1, iter_tensor)],
    )

    assert out_path.exists()

    with Image.open(out_path) as img:
        if hasattr(torch, "frombuffer"):
            data = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        else:
            data = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        arr = data.view(img.height, img.width)

    # We expect four panels: initial, intermediate, pred, gt.
    panel_w = arr.shape[1] // 4
    initial_panel = arr[:, :panel_w]
    iter_panel = arr[:, panel_w : 2 * panel_w]

    expected_initial = _expected_uint8(initial[0])
    expected_iter = _expected_uint8(iter_tensor[0])

    torch.testing.assert_close(initial_panel, expected_initial.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(iter_panel, expected_iter.cpu(), rtol=0, atol=0)


def test_save_side_by_side_pred_reuses_gt_normalization(tmp_path):
    height, width = 32, 32
    base = torch.linspace(0.0, 1.0, steps=height * width, dtype=torch.float32).view(
        1, 1, height, width
    )
    pred = base.clone()
    gt = base * 2.0 + 0.5
    initial = torch.zeros(height, width, dtype=torch.float32)

    out_path = tmp_path / "viz_gt_norm.png"
    save_side_by_side(pred[0], gt[0], initial, str(out_path))

    with Image.open(out_path) as img:
        if hasattr(torch, "frombuffer"):
            data = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
        else:
            data = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        arr = data.view(img.height, img.width)

    panel_w = arr.shape[1] // 3
    pred_panel = arr[:, panel_w : 2 * panel_w]
    gt_panel = arr[:, 2 * panel_w : 3 * panel_w]

    gt_range = _normalization_range(gt[0, 0])
    expected_pred = _expected_uint8(pred[0, 0], normalization=gt_range)
    expected_gt = _expected_uint8(gt[0, 0], normalization=gt_range)

    pred_diff = pred_panel != expected_pred.cpu()
    gt_diff = gt_panel != expected_gt.cpu()

    assert pred_diff.any(), "Expected annotation to modify prediction panel"
    assert gt_diff.any(), "Expected annotation to modify ground-truth panel"

    assert torch.any(~pred_diff), "Normalization drifted across prediction panel"
    assert torch.any(~gt_diff), "Normalization drifted across ground-truth panel"

    assert pred_panel[-1, -1] == expected_pred[-1, -1]
    assert gt_panel[-1, -1] == expected_gt[-1, -1]
