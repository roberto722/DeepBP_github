#!/usr/bin/env python3
"""Run inference over a dataset split and optionally export outputs."""
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from dataset import HDF5Dataset, VOCDataset
from deepbp.config import TrainConfig, create_model
from deepbp.metrics import psnr, ssim
from deepbp.visualization import save_side_by_side


def _parse_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(item) for item in raw.split(",") if item.strip()]


def _apply_config_dict(cfg: TrainConfig, config_dict: Dict) -> None:
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _load_checkpoint_config(cfg: TrainConfig, checkpoint: Dict) -> None:
    ckpt_cfg = checkpoint.get("config")
    if not isinstance(ckpt_cfg, dict):
        return
    _apply_config_dict(cfg, ckpt_cfg)


def _load_config_json(cfg: TrainConfig, config_path: str) -> None:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    _apply_config_dict(cfg, config_dict)


def _build_dataset(cfg: TrainConfig, split: str):
    dataset_type = cfg.dataset_type.lower()
    input_dir = os.path.join(cfg.data_root, cfg.sino_dir)
    if dataset_type == "hdf5":
        target_dir = os.path.join(cfg.data_root, cfg.recs_dir)
        return HDF5Dataset(
            input_dir,
            target_dir,
            cfg.sino_min,
            cfg.sino_max,
            cfg.img_min,
            cfg.img_max,
            split=split,
            wavelength=cfg.wavelength,
            target_shape=(cfg.n_det, cfg.n_t),
        )
    if dataset_type == "voc":
        return VOCDataset(
            input_dir,
            cfg.sino_min,
            cfg.sino_max,
            cfg.img_min,
            cfg.img_max,
            split=split,
            target_shape=(cfg.n_det, cfg.n_t),
        )
    supported = ("hdf5", "voc")
    raise ValueError(
        f"Unsupported dataset_type '{cfg.dataset_type}'. Expected one of: {', '.join(supported)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--data-root", default=None, help="Override TrainConfig.data_root.")
    parser.add_argument("--sino-dir", default=None, help="Override TrainConfig.sino_dir.")
    parser.add_argument("--recs-dir", default=None, help="Override TrainConfig.recs_dir (HDF5 only).")
    parser.add_argument("--dataset-type", default=None, choices=["hdf5", "voc"], help="Dataset backend.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size per forward pass.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples.")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda, cpu).")
    parser.add_argument("--output-dir", default="./inference_outputs", help="Output directory.")
    parser.add_argument("--save-png", action="store_true", help="Save side-by-side PNGs.")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to a saved config.json from a training run.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Training run directory containing config.json (alternative to --config-path).",
    )
    parser.add_argument(
        "--intermediate-indices",
        default=None,
        help="Comma-separated indices of intermediate steps to include in PNGs.",
    )
    parser.add_argument(
        "--no-checkpoint-config",
        action="store_true",
        help="Ignore the configuration stored in the checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = TrainConfig()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.run_dir is not None and args.config_path is not None:
        raise ValueError("Use either --run-dir or --config-path, not both.")
    if args.run_dir is not None:
        config_path = os.path.join(args.run_dir, "config.json")
        _load_config_json(cfg, config_path)
    elif args.config_path is not None:
        _load_config_json(cfg, args.config_path)
    elif not args.no_checkpoint_config:
        _load_checkpoint_config(cfg, checkpoint)

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.sino_dir is not None:
        cfg.sino_dir = args.sino_dir
    if args.recs_dir is not None:
        cfg.recs_dir = args.recs_dir
    if args.dataset_type is not None:
        cfg.dataset_type = args.dataset_type

    model = create_model(cfg, device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.eval()

    dataset = _build_dataset(cfg, args.split)
    total = len(dataset)
    limit = min(total, args.max_samples) if args.max_samples is not None else total

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    per_sample: List[Dict[str, float]] = []
    indices = _parse_indices(args.intermediate_indices)

    with torch.no_grad():
        for start in tqdm(range(0, limit, args.batch), desc="Inference"):
            end = min(start + args.batch, limit)
            batch_sino = []
            batch_target = []
            batch_names = []
            for idx in range(start, end):
                sino, target = dataset[idx]
                batch_sino.append(sino)
                batch_target.append(target)
                path = dataset.file_list[idx]
                stem, _ = os.path.splitext(os.path.basename(path))
                batch_names.append(stem)

            sino_tensor = torch.stack(batch_sino, dim=0).to(device)
            target_tensor = torch.stack(batch_target, dim=0).to(device)

            pred, initial, intermediates = model(sino_tensor)
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

            batch_psnr = psnr(pred, target_tensor)
            batch_ssim = ssim(pred, target_tensor)
            batch_l1 = torch.mean(torch.abs(pred - target_tensor), dim=(1, 2, 3))

            for b, name in enumerate(batch_names):
                record = {
                    "name": name,
                    "psnr": float(batch_psnr[b].item()),
                    "ssim": float(batch_ssim[b].item()),
                    "l1": float(batch_l1[b].item()),
                }
                per_sample.append(record)

                out_npy = os.path.join(output_dir, f"{name}_pred.npy")
                np.save(out_npy, pred[b].detach().cpu().numpy())

                if args.save_png:
                    debug_steps: Optional[List[tuple]] = None
                    if indices and iter_sequence:
                        total_steps = len(iter_sequence)
                        selected: List[tuple] = []
                        seen_steps = set()
                        for idx in indices:
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
                    out_png = os.path.join(output_dir, f"{name}_pred.png")
                    save_side_by_side(
                        pred[b],
                        target_tensor[b],
                        initial_panel,
                        out_png,
                        vmin=None,
                        vmax=None,
                        iter_steps=debug_steps,
                    )

    if per_sample:
        avg_psnr = sum(r["psnr"] for r in per_sample) / len(per_sample)
        avg_ssim = sum(r["ssim"] for r in per_sample) / len(per_sample)
        avg_l1 = sum(r["l1"] for r in per_sample) / len(per_sample)
    else:
        avg_psnr = avg_ssim = avg_l1 = 0.0

    summary = {
        "split": args.split,
        "samples": len(per_sample),
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "l1": avg_l1,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": per_sample}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
