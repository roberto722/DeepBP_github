"""Compute reconstruction metrics between NIfTI volumes.

Update the configuration in the ``__main__`` section before executing the
script instead of relying on command-line arguments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import nibabel as nib
import numpy as np


@dataclass
class MetricResult:
    """Metrics computed for a single prediction/target pair."""

    name: str
    mse: float
    mae: float
    psnr: float
    ssim: float


def _strip_all_suffixes(path: Path) -> str:
    """Return the filename without any suffix (handles .nii.gz correctly)."""

    suffix = "".join(path.suffixes)
    if not suffix:
        return path.name
    return path.name[: -len(suffix)]


def _load_nifti_array(path: Path) -> np.ndarray:
    """Load a NIfTI file as a float64 numpy array."""

    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> MetricResult:
    """Compute MSE, MAE, PSNR and SSIM for two arrays."""

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch between prediction {pred.shape} and target {target.shape}")

    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    data_min = float(np.min([pred.min(), target.min()]))
    data_max = float(np.max([pred.max(), target.max()]))
    data_range = max(data_max - data_min, 1e-8)

    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((data_range ** 2) / mse))

    mu_x = float(pred.mean())
    mu_y = float(target.mean())
    sigma_x2 = float(pred.var())
    sigma_y2 = float(target.var())
    sigma_xy = float(np.mean((pred - mu_x) * (target - mu_y)))

    L = data_range
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = float(numerator / (denominator + 1e-12))

    return MetricResult("", mse, mae, psnr, ssim)


def _iter_prediction_files(
    directory: Path, extensions: Optional[Sequence[str]]
) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if not extensions:
            yield path
            continue
        if any(path.name.endswith(ext) for ext in extensions):
            yield path


def compute_metrics(
    predictions_dir: Path,
    targets_dir: Path,
    target_suffix: str,
    extensions: Optional[Sequence[str]] = (".nii", ".nii.gz"),
) -> List[MetricResult]:
    """Compute metrics for all NIfTI pairs in the provided directories."""

    results: List[MetricResult] = []

    for pred_path in _iter_prediction_files(predictions_dir, extensions):
        base = _strip_all_suffixes(pred_path)
        suffix = "".join(pred_path.suffixes)
        target_name = f"{base}{target_suffix}{suffix}"
        target_path = targets_dir / target_name

        if not target_path.exists():
            raise FileNotFoundError(
                f"Missing target file for prediction '{pred_path.name}': expected '{target_name}'"
            )

        pred = _load_nifti_array(pred_path)
        target = _load_nifti_array(target_path)
        metrics = _compute_metrics(pred, target)
        metrics.name = pred_path.name
        results.append(metrics)

    if not results:
        raise RuntimeError("No prediction files found to process.")

    return results


def save_csv(results: Sequence[MetricResult], destination: Path) -> None:
    with destination.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "mse", "mae", "psnr", "ssim"])
        for res in results:
            writer.writerow([res.name, res.mse, res.mae, res.psnr, res.ssim])

        summary = summarize_metrics(results)
        writer.writerow([summary.name, summary.mse, summary.mae, summary.psnr, summary.ssim])


def summarize_metrics(results: Sequence[MetricResult]) -> MetricResult:
    summary = MetricResult(
        name="summary",
        mse=float(np.mean([res.mse for res in results])),
        mae=float(np.mean([res.mae for res in results])),
        psnr=float(np.mean([res.psnr for res in results])),
        ssim=float(np.mean([res.ssim for res in results])),
    )
    return summary


def main(
    predictions_dir: Path,
    targets_dir: Path,
    target_suffix: str,
    extensions: Optional[Sequence[str]] = (".nii", ".nii.gz"),
    csv_path: Optional[Path] = None,
) -> None:
    if not predictions_dir.is_dir():
        raise NotADirectoryError(f"Predictions directory '{predictions_dir}' does not exist")
    if not targets_dir.is_dir():
        raise NotADirectoryError(f"Targets directory '{targets_dir}' does not exist")

    results = compute_metrics(predictions_dir, targets_dir, target_suffix, extensions)

    print("file,mse,mae,psnr,ssim")
    for res in results:
        print(f"{res.name},{res.mse},{res.mae},{res.psnr},{res.ssim}")

    summary = summarize_metrics(results)
    print(f"{summary.name},{summary.mse},{summary.mae},{summary.psnr},{summary.ssim}")

    if csv_path is not None:
        save_csv(results, csv_path)
        print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    # Update the paths and suffix below before running the script.
    PREDICTIONS_DIR = Path("/path/to/predictions")
    TARGETS_DIR = Path("/path/to/targets")
    TARGET_SUFFIX = "_target"
    EXTENSIONS = (".nii", ".nii.gz")  # Set to None to process every file
    CSV_PATH = None  # Optionally set to Path("metrics.csv")

    main(
        predictions_dir=PREDICTIONS_DIR,
        targets_dir=TARGETS_DIR,
        target_suffix=TARGET_SUFFIX,
        extensions=EXTENSIONS,
        csv_path=CSV_PATH,
    )
