#!/usr/bin/env python3
"""Run model inference on the test split and export predictions as NIfTI images."""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise ImportError("NumPy is required to run infer_test_set.py. Install it with 'pip install numpy'.") from exc

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise ImportError(
        "PyTorch is required to run infer_test_set.py. Install it with 'pip install torch'."
    ) from exc

from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - dependency missing at runtime
    nib = None

from dataset import load_hdf5_sample, load_nifti_sample
from deepbp.config import TrainConfig, create_model
from deepbp.inference import run_inference_steps


def _load_config(config_path: Optional[str]) -> TrainConfig:
    """Instantiate a :class:`TrainConfig` optionally overriding values from JSON."""

    cfg = TrainConfig()
    if not config_path:
        return cfg

    with open(config_path, "r") as f:
        data = json.load(f)

    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


class _TestDataset(Dataset):
    """Dataset wrapper that mirrors training preprocessing without targets."""

    def __init__(
        self,
        cfg: TrainConfig,
        split: str,
        require_target: bool = False,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.require_target = require_target

        dataset_type = cfg.dataset_type.lower()
        input_dir = os.path.join(cfg.data_root, cfg.sino_dir)
        target_shape = (cfg.n_det, cfg.n_t)

        if dataset_type == "hdf5":
            self.file_list = self._collect_hdf5_files(input_dir)
            self._loader = lambda path: load_hdf5_sample(
                input_path=path,
                target_dir=os.path.join(cfg.data_root, cfg.recs_dir),
                wavelength=str(cfg.wavelength),
                target_shape=target_shape,
                sino_min=cfg.sino_min,
                sino_max=cfg.sino_max,
                img_min=cfg.img_min,
                img_max=cfg.img_max,
                apply_normalization=True,
                normalize_target=cfg.normalize_targets,
                require_target=require_target,
            )
        elif dataset_type == "voc":
            self.file_list = self._collect_voc_files(input_dir)
            self._loader = lambda path: load_nifti_sample(
                input_path=path,
                target_dir=self._matching_voc_target(path),
                target_shape=target_shape,
                sino_min=cfg.sino_min,
                sino_max=cfg.sino_max,
                img_min=cfg.img_min,
                img_max=cfg.img_max,
                apply_normalization=True,
                normalize_target=cfg.normalize_targets,
                require_target=require_target,
            )
        else:
            raise ValueError(
                "Unsupported dataset_type '%s'. Expected 'hdf5' or 'voc'." % cfg.dataset_type
            )

    def _collect_hdf5_files(self, input_dir: str) -> List[str]:
        test_dir = os.path.join(input_dir, self.split)
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(
                f"Cannot locate '{self.split}' split under '{input_dir}'."
            )
        return sorted(
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.endswith(".hdf5")
        )

    def _collect_voc_files(self, input_dir: str) -> List[str]:
        if self.split == "test":
            base = os.path.join(input_dir, "tst", self.split, "sinograms")
        else:
            base = os.path.join(input_dir, "trn_val", self.split, "sinograms")
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"Cannot locate '{self.split}' split under '{input_dir}'."
            )
        return sorted(
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.endswith(".nii")
        )

    @staticmethod
    def _matching_voc_target(sino_path: str) -> str:
        return (
            sino_path.replace("_sinogram", "_rec_img_L1_shearlet_e-05")
            .replace("sinograms", "rec_images")
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.file_list[idx]
        sinogram, _ = self._loader(path)
        return sinogram, path


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _denormalize_prediction(pred: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if not cfg.normalize_targets:
        return pred
    scale = cfg.img_max - cfg.img_min
    return pred * scale + cfg.img_min


def _save_prediction(
    image: torch.Tensor,
    reference_path: str,
    output_dir: Path,
    squeeze: bool = True,
) -> None:
    if nib is None:
        raise ImportError(
            "nibabel is required to save predictions in NIfTI format."
            " Install it with 'pip install nibabel'."
        )
    array = image.detach().cpu().to(dtype=torch.float32)
    if squeeze:
        array = array.squeeze()
    array_np = array.numpy().astype(np.float32, copy=False)
    nifti = nib.Nifti1Image(array_np, affine=np.eye(4))
    name = Path(reference_path).name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = Path(name).stem
    output_path = output_dir / f"{stem}_prediction.nii.gz"
    nib.save(nifti, str(output_path))


def run_inference(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = create_model(cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint missing 'model' key. Expected training checkpoint with model state_dict."
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = _TestDataset(cfg, split=args.split, require_target=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    output_dir = _ensure_output_dir(Path(args.output_dir))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            sinograms, paths = batch
            sinograms = sinograms.to(device)
            for sino, ref_path in zip(sinograms, paths):
                _, _, pred, _, _ = run_inference_steps(
                    model,
                    sino,
                    cfg,
                    device=device,
                    normalize=False,
                )
                pred = _denormalize_prediction(pred.squeeze(0), cfg)
                _save_prediction(pred, ref_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint (expects a 'model' state_dict entry).",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional path to the JSON configuration exported during training. "
            "If omitted, defaults from TrainConfig are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where NIfTI predictions will be saved.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to process (default: test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference dataloader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for the dataloader.",
    )
    parser.add_argument(
        "--device",
        help="Computation device override (e.g. 'cuda:0' or 'cpu').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
