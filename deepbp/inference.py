"""Utilities for iterative inference and visualization."""
from typing import List, Optional, Tuple

import torch

from dataset import minmax_scale
from .config import TrainConfig


def _extract_canonical_initial(
    beamformer_output: torch.Tensor, reference: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return a view containing the first channel/batch element for display."""

    x = beamformer_output
    if reference is not None and reference.dim() >= 1:
        ref_batch = reference.shape[0]
    else:
        ref_batch = None

    if x.dim() == 4:
        # (B, C, H, W)
        return x[:, :1, ...]

    if x.dim() == 3:
        # Ambiguous: could be (B, H, W) or (C, H, W) with B == 1.
        if ref_batch is not None and x.shape[0] == ref_batch:
            return x.unsqueeze(1)
        return x[:1, ...].unsqueeze(0)

    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)

    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)

    return x


def run_inference_steps(
    model: torch.nn.Module,
    sinogram: torch.Tensor,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[List[torch.Tensor]],
    Optional[torch.Tensor],
]:
    """Return normalized sinogram, initial image, final output, intermediates and beamformer stack."""

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
    canonical_initial: Optional[torch.Tensor] = None
    if beamformer_img is not None:
        canonical_initial = _extract_canonical_initial(beamformer_img, pred)
        iter_sequence.append(canonical_initial)

    if intermediates is not None:
        if isinstance(intermediates, (list, tuple)):
            iter_sequence.extend(intermediates)
        else:
            iter_sequence.append(intermediates)

    if pred is not None and (not iter_sequence or iter_sequence[-1] is not pred):
        iter_sequence.append(pred)

    iter_imgs = [step.detach().cpu() for step in iter_sequence] if iter_sequence else None

    beamformer_cpu = (
        canonical_initial.detach().cpu() if canonical_initial is not None else None
    )
    beamformer_stack_cpu = (
        beamformer_img.detach().cpu() if beamformer_img is not None else None
    )

    return (
        sino_norm.detach().cpu(),
        beamformer_cpu,
        pred.detach().cpu(),
        iter_imgs,
        beamformer_stack_cpu,
    )
