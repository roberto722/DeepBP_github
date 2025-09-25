"""Helpers to export side-by-side inspection images."""
import math
import os
from typing import List, Optional, Tuple

import torch
from PIL import Image


def save_side_by_side(
    pred: torch.Tensor,
    gt: torch.Tensor,
    initial: torch.Tensor,
    out_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    iter_steps: Optional[List[Tuple[int, torch.Tensor]]] = None,
) -> None:
    """Save a side-by-side image [Initial | intermediates | Pred | GT]."""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def to_uint8(img: torch.Tensor) -> Image.Image:
        x = img.detach()

        if x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 1:
            x = x[0, 0]
        elif x.dim() == 3 and x.shape[0] == 1:
            x = x[0]
        else:
            x = x.squeeze()

        if x.dim() < 2:
            raise ValueError(f"Expected image-like tensor, got shape {tuple(img.shape)}")

        x = x.to(dtype=torch.float32)

        local_min = local_max = None
        if vmin is None or vmax is None:
            finite_mask = torch.isfinite(x)
            if finite_mask.any():
                valid = x[finite_mask]
                local_min, local_max = torch.aminmax(valid)
            else:
                local_min = torch.tensor(0.0, dtype=x.dtype, device=x.device)
                local_max = torch.tensor(1.0, dtype=x.dtype, device=x.device)

        lo = float(local_min.item()) if vmin is None and local_min is not None else float(vmin or 0.0)
        hi = float(local_max.item()) if vmax is None and local_max is not None else float(vmax or lo)

        if not math.isfinite(lo):
            lo = 0.0
        if not math.isfinite(hi):
            hi = lo
        if hi - lo < 1e-6:
            hi = lo + 1e-6

        x = torch.nan_to_num(x, nan=lo, posinf=hi, neginf=lo)
        x = (x - lo) / (hi - lo)
        x = x.clamp(0.0, 1.0)

        x = (x * 255.0).round().to(dtype=torch.uint8).cpu().numpy()
        return Image.fromarray(x, mode="L")

    panels: List[Image.Image] = []
    panels.append(to_uint8(initial))

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
