"""Evaluation metrics for reconstructions."""
from typing import Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute PSNR per-image assuming inputs in [0, 1] or similar scale."""

    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
    window: int = 11,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lightweight SSIM approximation on single-channel images."""

    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, 1, pad)
    mu_y = F.avg_pool2d(target, window, 1, pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(pred * pred, window, 1, pad) - mu_x2
    sigma_y2 = F.avg_pool2d(target * target, window, 1, pad) - mu_y2
    sigma_xy = F.avg_pool2d(pred * target, window, 1, pad) - mu_xy

    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    ssim_flat = ssim_map.flatten(1)

    if mask is None:
        return ssim_flat.mean(dim=1)

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape != ssim_map.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} incompatible with SSIM map shape {tuple(ssim_map.shape)}"
        )

    mask = mask.to(ssim_map.dtype)
    mask_flat = mask.flatten(1)
    masked_sum = (ssim_flat * mask_flat).sum(dim=1)
    mask_norm = mask_flat.sum(dim=1)
    global_mean = ssim_flat.mean(dim=1)
    masked_mean = masked_sum / mask_norm.clamp_min(1.0)
    return torch.where(mask_norm > 0, masked_mean, global_mean)
