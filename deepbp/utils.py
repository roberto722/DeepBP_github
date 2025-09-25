"""Generic utilities for reproducibility and weighting."""
import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


@torch.no_grad()
def compute_global_minmax_from_loader(loader, get_sino, get_img):
    """Compute per-domain global min/max across the whole training loader."""

    smin, smax = np.inf, -np.inf
    imin, imax = np.inf, -np.inf
    for batch in loader:
        sino = get_sino(batch)
        img = get_img(batch)
        sino_np = sino.detach().cpu().numpy()
        img_np = img.detach().cpu().numpy()
        smin = min(smin, np.nanmin(sino_np))
        smax = max(smax, np.nanmax(sino_np))
        imin = min(imin, np.nanmin(img_np))
        imax = max(imax, np.nanmax(img_np))
    return {"sino": {"min": float(smin), "max": float(smax)}, "img": {"min": float(imin), "max": float(imax)}}


def save_stats_json(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
