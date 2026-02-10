#!/usr/bin/env python3
"""Configurable folder inference script with raw export and reconstruction metrics.

Configure all parameters in the CONFIG dataclass below and run:
    python tools/inference_folder_config.py
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataset import load_hdf5_sample, load_nifti_sample
from deepbp.config import TrainConfig, create_model
from deepbp.metrics import psnr, ssim


@dataclass
class InferenceConfig:
    # --- Paths ---
    # If run_dir is set, configuration is loaded automatically from:
    # 1) <run_dir>/config.json
    # 2) fallback to checkpoint embedded config (best.pt/last.pt/first available .pt)
    run_dir: Optional[str] = "./checkpoints/experiment"
    checkpoint_path: Optional[str] = None
    input_dir: str = "./data/input"
    target_dir: str = "./data/target"
    output_dir: str = "./inference_raw_outputs"

    # --- Dataset parsing ---
    data_format: str = "hdf5"  # "hdf5" or "nii"
    wavelength: str = "800"  # used only for hdf5 target filename pattern
    input_extension: str = ".hdf5"  # ".hdf5" or ".nii"
    target_extension: str = ".nii"
    target_suffix_replace_from: Optional[str] = None  # e.g. "_sinogram"
    target_suffix_replace_to: Optional[str] = None  # e.g. "_rec_img_L1_shearlet_e-05"

    # --- Preprocessing / geometry ---
    target_shape: Tuple[int, int] = (128, 1640)
    sino_min: float = -0.10295463952521738
    sino_max: float = 0.07286326741859795
    img_min: float = 0.0
    img_max: float = 255.0

    # --- Runtime ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: Optional[int] = None


CONFIG = InferenceConfig()


def _apply_config_dict(cfg: TrainConfig, config_dict: Dict) -> None:
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _load_json_config(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config file content: {path}")
    return payload


def _auto_checkpoint_from_run_dir(run_dir: Path) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpt_paths = sorted(p for p in run_dir.glob("*.pt") if p.is_file())
    if not ckpt_paths:
        raise FileNotFoundError(f"No .pt checkpoint found in run directory: {run_dir}")

    by_name = {p.name: p for p in ckpt_paths}
    for preferred in ("best.pt", "last.pt"):
        if preferred in by_name:
            return by_name[preferred]
    return ckpt_paths[0]


def _load_experiment_config(train_cfg: TrainConfig, checkpoint: Dict, run_dir: Optional[Path]) -> str:
    if run_dir is not None:
        run_cfg_path = run_dir / "config.json"
        file_cfg = _load_json_config(run_cfg_path)
        if file_cfg is not None:
            _apply_config_dict(train_cfg, file_cfg)
            return f"file:{run_cfg_path}"

    ckpt_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(ckpt_cfg, dict):
        _apply_config_dict(train_cfg, ckpt_cfg)
        return "checkpoint:embedded"

    return "default"


def _resolve_target_path(input_path: str, cfg: InferenceConfig) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]

    if cfg.data_format == "hdf5":
        # Match existing dataset convention
        filename = f"{stem}_{cfg.wavelength}_rec_L1_shearlet{cfg.target_extension}"
        return os.path.join(cfg.target_dir, filename)

    target_stem = stem
    if cfg.target_suffix_replace_from is not None and cfg.target_suffix_replace_to is not None:
        target_stem = target_stem.replace(cfg.target_suffix_replace_from, cfg.target_suffix_replace_to)

    return os.path.join(cfg.target_dir, f"{target_stem}{cfg.target_extension}")


def _load_pair(input_path: str, cfg: InferenceConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.data_format == "hdf5":
        return load_hdf5_sample(
            input_path=input_path,
            target_dir=cfg.target_dir,
            wavelength=cfg.wavelength,
            target_shape=cfg.target_shape,
            sino_min=cfg.sino_min,
            sino_max=cfg.sino_max,
            img_min=cfg.img_min,
            img_max=cfg.img_max,
            apply_normalization=True,
            require_target=True,
        )

    if cfg.data_format == "nii":
        target_path = _resolve_target_path(input_path, cfg)
        return load_nifti_sample(
            input_path=input_path,
            target_dir=target_path,
            target_shape=cfg.target_shape,
            sino_min=cfg.sino_min,
            sino_max=cfg.sino_max,
            img_min=cfg.img_min,
            img_max=cfg.img_max,
            apply_normalization=True,
            require_target=True,
        )

    raise ValueError("CONFIG.data_format must be either 'hdf5' or 'nii'.")


def _save_raw(path: str, tensor: torch.Tensor) -> Dict[str, object]:
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(path)
    return {
        "path": path,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def main() -> None:
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    pred_raw_dir = os.path.join(CONFIG.output_dir, "pred_raw")
    gt_raw_dir = os.path.join(CONFIG.output_dir, "gt_raw")
    os.makedirs(pred_raw_dir, exist_ok=True)
    os.makedirs(gt_raw_dir, exist_ok=True)

    input_files = sorted(
        os.path.join(CONFIG.input_dir, name)
        for name in os.listdir(CONFIG.input_dir)
        if name.endswith(CONFIG.input_extension)
    )

    if CONFIG.max_samples is not None:
        input_files = input_files[: CONFIG.max_samples]

    if not input_files:
        raise RuntimeError(f"No files with extension '{CONFIG.input_extension}' found in {CONFIG.input_dir}")

    device = torch.device(CONFIG.device)
    train_cfg = TrainConfig()

    run_dir = Path(CONFIG.run_dir).resolve() if CONFIG.run_dir else None
    if CONFIG.checkpoint_path is not None:
        checkpoint_path = Path(CONFIG.checkpoint_path).resolve()
    elif run_dir is not None:
        checkpoint_path = _auto_checkpoint_from_run_dir(run_dir)
    else:
        raise ValueError("Set CONFIG.checkpoint_path or CONFIG.run_dir.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_source = _load_experiment_config(train_cfg, checkpoint, run_dir)

    model = create_model(train_cfg, device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.eval()

    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for input_path in input_files:
            name = os.path.splitext(os.path.basename(input_path))[0]
            sinogram, target = _load_pair(input_path, CONFIG)

            sinogram = sinogram.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            pred, _, _ = model(sinogram)

            sample_psnr = float(psnr(pred, target)[0].item())
            sample_ssim = float(ssim(pred, target)[0].item())
            sample_mse = float(torch.mean((pred - target) ** 2).item())
            sample_mae = float(torch.mean(torch.abs(pred - target)).item())

            pred_raw = _save_raw(os.path.join(pred_raw_dir, f"{name}.raw"), pred[0])
            gt_raw = _save_raw(os.path.join(gt_raw_dir, f"{name}.raw"), target[0])

            records.append(
                {
                    "name": name,
                    "input_path": input_path,
                    "target_path": _resolve_target_path(input_path, CONFIG),
                    "pred_raw": pred_raw,
                    "gt_raw": gt_raw,
                    "metrics": {
                        "ssim": sample_ssim,
                        "psnr": sample_psnr,
                        "mse": sample_mse,
                        "mae": sample_mae,
                    },
                }
            )

    summary = {
        "samples": len(records),
        "checkpoint": str(checkpoint_path),
        "config_source": config_source,
        "ssim": float(np.mean([r["metrics"]["ssim"] for r in records])),
        "psnr": float(np.mean([r["metrics"]["psnr"] for r in records])),
        "mse": float(np.mean([r["metrics"]["mse"] for r in records])),
        "mae": float(np.mean([r["metrics"]["mae"] for r in records])),
    }

    with open(os.path.join(CONFIG.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
