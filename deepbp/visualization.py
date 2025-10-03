"""Helpers to export side-by-side inspection images."""
import math
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


def save_side_by_side(
    pred: torch.Tensor,
    gt: torch.Tensor,
    initial: torch.Tensor,
    out_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    iter_steps: Optional[List[Tuple[int, torch.Tensor]]] = None,
) -> None:
    """Save a side-by-side image [Initial | intermediates | Pred | GT].

    When tensors contain multiple channels, the first channel is used for visualization.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _prepare_for_display(img: torch.Tensor) -> torch.Tensor:
        x = img.detach()

        while x.dim() > 2:
            if x.size(0) == 0:
                raise ValueError("Cannot visualize an empty tensor")
            x = x[0]

        if x.dim() < 2:
            raise ValueError(f"Expected image-like tensor, got shape {tuple(img.shape)}")

        return x.to(dtype=torch.float32)

    def _compute_normalization(
        x: torch.Tensor,
        *,
        min_override: Optional[float] = None,
        max_override: Optional[float] = None,
    ) -> Tuple[float, float]:
        local_min = local_max = None
        if min_override is None or max_override is None:
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                valid = x[finite_mask]
                local_min, local_max = torch.aminmax(valid)
            else:
                local_min = torch.tensor(0.0, dtype=x.dtype, device=x.device)
                local_max = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        if min_override is not None:
            lo = float(min_override)
        elif local_min is not None:
            lo = float(local_min.item())
        else:
            lo = 0.0

        if max_override is not None:
            hi = float(max_override)
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

    def to_uint8(
        img: torch.Tensor,
        *,
        normalization: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        x = _prepare_for_display(img)

        if normalization is None:
            lo, hi = _compute_normalization(
                x, min_override=vmin, max_override=vmax
            )
        else:
            lo, hi = _compute_normalization(
                x, min_override=normalization[0], max_override=normalization[1]
            )

        x = torch.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
        x = (x - lo) / (hi - lo)
        x = x.clamp(0.0, 1.0)

        x = (x * 255.0).round().to(dtype=torch.uint8).cpu().numpy()
        return Image.fromarray(x, mode="L")

    def _annotate_panel(panel: Image.Image, label: str) -> Image.Image:
        annotated = panel.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.load_default()
        except OSError:
            font = None

        margin = max(1, min(4, annotated.width // 20, annotated.height // 20))
        if margin <= 0:
            margin = 1
        position = (margin, margin)
        outline = 1
        text_kwargs = {"font": font} if font is not None else {}

        if outline > 0:
            for dx in range(-outline, outline + 1):
                for dy in range(-outline, outline + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text(
                        (position[0] + dx, position[1] + dy),
                        label,
                        fill=0,
                        **text_kwargs,
                    )

        draw.text(position, label, fill=255, **text_kwargs)
        return annotated

    gt_normalization: Optional[Tuple[float, float]] = None
    if gt.numel() > 0:
        gt_tensor = _prepare_for_display(gt)
        gt_normalization = _compute_normalization(
            gt_tensor, min_override=vmin, max_override=vmax
        )

    panels: List[Image.Image] = []
    panels.append(to_uint8(initial))

    if iter_steps:
        seen = set()
        for idx, tensor in sorted(iter_steps, key=lambda item: item[0]):
            if idx in seen:
                continue
            seen.add(idx)
            panels.append(to_uint8(tensor))

    pred_panel = (
        to_uint8(pred, normalization=gt_normalization) if gt_normalization else to_uint8(pred)
    )
    gt_panel = (
        to_uint8(gt, normalization=gt_normalization) if gt_normalization else to_uint8(gt)
    )

    panels.append(_annotate_panel(pred_panel, "Prediction"))
    panels.append(_annotate_panel(gt_panel, "Ground Truth"))

    width = sum(im.width for im in panels)
    height = max(im.height for im in panels)
    canvas = Image.new("L", (width, height))
    xoff = 0
    for im in panels:
        canvas.paste(im, (xoff, 0))
        xoff += im.width
    canvas.save(out_path)
