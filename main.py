# -*- coding: utf-8 -*-
"""Entry-point script wiring datasets, model and training loops."""
import json
import os
import shutil
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from dataset import HDF5Dataset
from deepbp.config import TrainConfig, create_model
from deepbp.training import train_one_epoch, validate
from deepbp.utils import seed_everything


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Instantiate training and validation dataloaders based on the configuration."""

    input_dir = os.path.join(cfg.data_root, cfg.sino_dir)
    target_dir = os.path.join(cfg.data_root, cfg.recs_dir)

    train_ds = HDF5Dataset(
        input_dir,
        target_dir,
        cfg.sino_min,
        cfg.sino_max,
        cfg.img_min,
        cfg.img_max,
        split="train",
        wavelength=cfg.wavelength,
        target_shape=(cfg.n_det, cfg.n_t),
    )
    val_ds = HDF5Dataset(
        input_dir,
        target_dir,
        cfg.sino_min,
        cfg.sino_max,
        cfg.img_min,
        cfg.img_max,
        split="val",
        wavelength=cfg.wavelength,
        target_shape=(cfg.n_det, cfg.n_t),
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_ld, val_ld


def main() -> None:
    seed_everything(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    if os.path.exists(cfg.work_dir):
        shutil.rmtree(cfg.work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    img_dir = os.path.join(cfg.work_dir, "val_images")
    os.makedirs(img_dir, exist_ok=True)

    model = create_model(cfg, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model configuration.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_ld, val_ld = build_dataloaders(cfg)

    with open(os.path.join(cfg.work_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    best_psnr = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model,
            train_ld,
            optimizer,
            device,
            clip_grad=cfg.clip_grad,
            weight_alpha=cfg.weight_alpha,
            weight_threshold=cfg.weight_threshold,
            use_tqdm=cfg.use_tqdm,
        )
        val = validate(
            model,
            val_ld,
            device,
            epoch,
            save_dir=img_dir if cfg.save_val_images else None,
            max_save=cfg.max_val_images,
            intermediate_indices=cfg.val_intermediate_indices,
            weight_alpha=cfg.weight_alpha,
            weight_threshold=cfg.weight_threshold,
            ssim_mask_threshold=cfg.ssim_mask_threshold,
            ssim_mask_dilation=cfg.ssim_mask_dilation,
            use_tqdm=cfg.use_tqdm,
        )

        scheduler.step()

        log = {
            "epoch": epoch,
            "train_psnr": tr["psnr"],
            "train_ssim": tr["ssim"],
            "train_l1": tr["l1"],
            "train_weighted_l1": tr["weighted_l1"],
            "val_psnr": val["psnr"],
            "val_ssim": val["ssim"],
            "val_l1": val["l1"],
            "val_weighted_l1": val["weighted_l1"],
            "val_masked_ssim": val.get("masked_ssim"),
            "lr": scheduler.get_last_lr()[0],
        }
        print(json.dumps(log, indent=2))

        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_psnr": best_psnr,
                    "config": cfg.__dict__,
                },
                os.path.join(cfg.work_dir, "best.pt"),
            )

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": cfg.epochs,
            "val_psnr": best_psnr,
            "config": cfg.__dict__,
        },
        os.path.join(cfg.work_dir, "last.pt"),
    )


if __name__ == "__main__":
    main()
