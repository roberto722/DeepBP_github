"""Training and validation loops."""
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import psnr, ssim
from .utils import build_bright_mask, compute_intensity_weights
from .visualization import save_side_by_side


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    save_dir: Optional[str] = None,
    max_save: int = 8,
    intermediate_indices: Optional[List[int]] = None,
    weight_alpha: float = 0.0,
    weight_threshold: Optional[float] = None,
    ssim_mask_threshold: Optional[float] = None,
    ssim_mask_dilation: int = 0,
    use_tqdm: bool = True,
) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0, "weighted_l1": 0.0}
    compute_masked_ssim = ssim_mask_threshold is not None
    if compute_masked_ssim:
        agg["masked_ssim"] = 0.0
    n = 0
    saved = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="Val") if use_tqdm else None
        iterator = progress if progress is not None else loader
        for i, (sino, img) in enumerate(iterator):
            sino = sino.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)

            pred, initial, intermediates = model(sino)
            iter_sequence: List[torch.Tensor] = []
            if initial is not None:
                iter_sequence.append(initial)
            if intermediates is not None:
                if isinstance(intermediates, (list, tuple)):
                    iter_sequence.extend(intermediates)
                else:
                    iter_sequence.append(intermediates)
            if pred is not None and (not iter_sequence or iter_sequence[-1] is not pred):
                iter_sequence.append(pred)

            batch_psnr = psnr(pred, img).mean()
            batch_ssim = ssim(pred, img).mean()
            batch_l1 = F.l1_loss(pred, img)
            weights = compute_intensity_weights(img, weight_alpha, weight_threshold)
            batch_weighted_l1 = torch.mean(weights * torch.abs(pred - img))

            if compute_masked_ssim:
                mask = build_bright_mask(img, ssim_mask_threshold, dilation=ssim_mask_dilation)
                batch_masked_ssim = ssim(pred, img, mask=mask).mean()

            bs = sino.size(0)
            n += bs
            agg["psnr"] += batch_psnr.item() * bs
            agg["ssim"] += batch_ssim.item() * bs
            agg["l1"] += batch_l1.item() * bs
            agg["weighted_l1"] += batch_weighted_l1.item() * bs
            if compute_masked_ssim:
                agg["masked_ssim"] += batch_masked_ssim.item() * bs

            if progress is not None:
                denom = max(n, 1)
                postfix = {
                    "psnr": f"{agg['psnr'] / denom:.4f}",
                    "ssim": f"{agg['ssim'] / denom:.4f}",
                    "l1": f"{agg['l1'] / denom:.4f}",
                    "w_l1": f"{agg['weighted_l1'] / denom:.4f}",
                }
                if compute_masked_ssim:
                    postfix["masked_ssim"] = f"{agg['masked_ssim'] / denom:.4f}"
                progress.set_postfix(postfix)

            if save_dir is not None and saved < max_save:
                for b in range(min(bs, max_save - saved)):
                    out_path = os.path.join(save_dir, f"val_epoch_{epoch}_{i:04d}_{b:02d}.png")
                    debug_steps: Optional[List[Tuple[int, torch.Tensor]]] = None
                    if intermediate_indices and iter_sequence:
                        total_steps = len(iter_sequence)
                        selected: List[Tuple[int, torch.Tensor]] = []
                        seen_steps = set()
                        for idx in intermediate_indices:
                            actual_idx = idx if idx >= 0 else total_steps + idx
                            if actual_idx < 0 or actual_idx >= total_steps:
                                continue
                            if actual_idx in seen_steps:
                                continue
                            seen_steps.add(actual_idx)
                            if actual_idx == 0 or actual_idx == total_steps - 1:
                                continue
                            selected.append((actual_idx, iter_sequence[actual_idx][b]))
                        if selected:
                            selected.sort(key=lambda item: item[0])
                            debug_steps = selected
                    initial_panel = initial[b] if initial is not None else pred[b]
                    save_side_by_side(
                        pred[b],
                        img[b],
                        initial_panel,
                        out_path,
                        vmin=None,
                        vmax=None,
                        iter_steps=debug_steps,
                    )

                    base_path, _ = os.path.splitext(out_path)

                    def _save_nifti(tensor: torch.Tensor, suffix: str) -> None:
                        array = tensor.detach().cpu().to(dtype=torch.float32).numpy()
                        nifti = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
                        nib.save(nifti, f"{base_path}_{suffix}.nii.gz")

                    _save_nifti(pred[b], "pred")
                    _save_nifti(img[b], "target")
                    _save_nifti(initial_panel, "initial")
                    saved += 1

        if progress is not None:
            progress.close()

    for k in agg:
        agg[k] /= max(n, 1)
    return agg


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: Optional[float] = 1.0,
    weight_alpha: float = 0.0,
    weight_threshold: Optional[float] = None,
    use_tqdm: bool = True,
) -> Dict[str, float]:
    model.train()
    agg: Dict[str, float] = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0, "weighted_l1": 0.0}
    n = 0
    progress = tqdm(loader, desc="Train") if use_tqdm else None
    iterator = progress if progress is not None else loader
    for _, (sino, img) in enumerate(iterator):
        sino = sino.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)

        pred, _, _ = model(sino)
        weights = compute_intensity_weights(img, weight_alpha, weight_threshold)
        loss_weighted_l1 = torch.mean(weights * torch.abs(pred - img))

        optimizer.zero_grad(set_to_none=True)
        loss_weighted_l1.backward()
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                clip_grad,
            )
        optimizer.step()

        with torch.no_grad():
            bs = sino.size(0)
            n += bs
            agg["weighted_l1"] += loss_weighted_l1.item() * bs
            agg["l1"] += F.l1_loss(pred, img).item() * bs
            agg["psnr"] += psnr(pred, img).mean().item() * bs
            agg["ssim"] += ssim(pred, img).mean().item() * bs

            if progress is not None:
                denom = max(n, 1)
                progress.set_postfix(
                    {
                        "psnr": f"{agg['psnr'] / denom:.4f}",
                        "ssim": f"{agg['ssim'] / denom:.4f}",
                        "l1": f"{agg['l1'] / denom:.4f}",
                        "w_l1": f"{agg['weighted_l1'] / denom:.4f}",
                    }
                )

    for k in agg:
        agg[k] /= max(n, 1)
    if progress is not None:
        progress.close()
    return agg
