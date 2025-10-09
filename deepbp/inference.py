"""Utilities for iterative inference and visualization."""
from typing import List, Optional, Tuple

import torch

from dataset import minmax_scale
from .config import TrainConfig


def run_inference_steps(
    model: torch.nn.Module,
    sinogram: torch.Tensor,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[List[torch.Tensor]]]:
    """Return normalized sinogram, initial image, final output and intermediate steps."""

    if device is None:
        device = next(model.parameters()).device

    if sinogram.dim() == 2:
        sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    elif sinogram.dim() == 3:
        if sinogram.shape[0] != 1 and sinogram.shape[1] != 1:
            sinogram = sinogram.unsqueeze(1)
        elif sinogram.shape[0] == 1:
            sinogram = sinogram.unsqueeze(0)
    elif sinogram.dim() != 4:
        raise ValueError(f"Unexpected sinogram shape: {tuple(sinogram.shape)}")

    dtype = next(model.parameters()).dtype
    sino = sinogram.to(device=device, dtype=dtype)
    sino_norm = minmax_scale(sino, cfg.sino_min, cfg.sino_max) if normalize else sino

    model.eval()
    with torch.no_grad():
        pred, beamformer_img, intermediates = model(sino_norm)

    iter_sequence: List[torch.Tensor] = []
    if beamformer_img is not None:
        iter_sequence.append(beamformer_img)

    if intermediates is not None:
        if isinstance(intermediates, (list, tuple)):
            iter_sequence.extend(intermediates)
        else:
            iter_sequence.append(intermediates)

    if pred is not None and (not iter_sequence or iter_sequence[-1] is not pred):
        iter_sequence.append(pred)

    iter_imgs = [step.detach().cpu() for step in iter_sequence] if iter_sequence else None

    beamformer_cpu = beamformer_img.detach().cpu() if beamformer_img is not None else None

    return (
        sino_norm.detach().cpu(),
        beamformer_cpu,
        pred.detach().cpu(),
        iter_imgs,
    )
