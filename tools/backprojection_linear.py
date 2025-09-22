#!/usr/bin/env python3
"""Command-line utility to run linear-probe backprojection on HDF5 sinograms.

The implementation mirrors the MATLAB routines ``backproject_waveeq_linear`` and
``bp_kernel`` provided by the customer, reusing the data loading logic from the
training pipeline when possible.  When PyTorch is not available the loader is
re-implemented in NumPy to keep the script lightweight and easy to execute in a
minimal environment.

Example usage::

    python tools/backprojection_linear.py \
        --input path/to/sample.hdf5 \
        --data-root /path/to/datasets \
        --sino-dir Forearm2000_hdf5/train_val_tst \
        --recs-dir Forearm2000_recs/L1_Shearlet \
        --split val \
        --output bp_result.npy

The script supports optional Gaussian pre-filtering, kernel caching, PNG export
and a compact summary of the reconstruction statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np

try:  # SciPy is required for the Hankel (Bessel) function used in the kernel.
    from scipy import special
except ImportError as exc:  # pragma: no cover - handled during runtime
    raise ImportError(
        "scipy is required to build the backprojection kernel. Install it with "
        "`pip install scipy`."
    ) from exc

try:  # Optional imports from the training code (require torch/nibabel at runtime)
    from dataset import load_hdf5_sample as torch_load_hdf5_sample
except ImportError:  # pragma: no cover - torch/nibabel may be unavailable on CI
    torch_load_hdf5_sample = None

try:
    from main import LinearProbeGeom as TorchLinearProbeGeom
    from main import TrainConfig as TorchTrainConfig
    from main import build_geometry as torch_build_geometry
except ImportError:  # pragma: no cover - PyTorch might be missing on CI
    TorchLinearProbeGeom = None
    TorchTrainConfig = None
    torch_build_geometry = None


# -----------------------------------------------------------------------------
# Minimal fallbacks when the training modules cannot be imported
# -----------------------------------------------------------------------------

@dataclass
class LinearProbeGeom:
    """Acquisition geometry for a 2-D linear ultrasound probe."""

    n_det: int
    pitch_m: float
    t0_s: float
    dt_s: float
    n_t: int
    c_m_s: float
    x0_m: float
    y0_m: float
    dx_m: float
    dy_m: float
    nx: int
    ny: int
    array_x0_m: float = 0.0
    array_y_m: float = 0.0

    @property
    def det_x(self) -> np.ndarray:
        """Return the detector *x* coordinates as a 1-D array (meters)."""

        xs = self.array_x0_m + np.arange(self.n_det, dtype=np.float64) * self.pitch_m
        return xs

    @property
    def det_y(self) -> np.ndarray:
        """Return the detector *y* coordinates as a 1-D array (meters)."""

        return np.full((self.n_det,), self.array_y_m, dtype=np.float64)


@dataclass
class TrainConfig:
    """Subset of :class:`main.TrainConfig` needed for geometry and data paths."""

    n_det: int = 128
    pitch_m: float = 0.0003
    t0_s: float = 0.0
    dt_s: float = 1 / 31.25e6
    n_t: int = 1640
    c_m_s: float = 1540.0
    x0_m: float = -0.019
    y0_m: float = 0.0
    dx_m: float = 0.00015
    dy_m: float = 0.00015
    nx: int = 256
    ny: int = 256
    array_x0_m: float = -0.019
    array_y_m: float = 0.0
    wavelength: int = 800
    sino_min: float = -11.0322
    sino_max: float = 12.5394
    img_min: float = 0.0
    img_max: float = 316.9658
    data_root: str = ""
    sino_dir: str = ""
    recs_dir: str = ""


def build_geometry(cfg: TrainConfig) -> LinearProbeGeom:
    """Return a :class:`LinearProbeGeom` populated from ``cfg``."""

    return LinearProbeGeom(
        n_det=cfg.n_det,
        pitch_m=cfg.pitch_m,
        t0_s=cfg.t0_s,
        dt_s=cfg.dt_s,
        n_t=cfg.n_t,
        c_m_s=cfg.c_m_s,
        x0_m=cfg.x0_m,
        y0_m=cfg.y0_m,
        dx_m=cfg.dx_m,
        dy_m=cfg.dy_m,
        nx=cfg.nx,
        ny=cfg.ny,
        array_x0_m=cfg.array_x0_m,
        array_y_m=cfg.array_y_m,
    )


# If the training modules are available we reuse the original definitions
if TorchTrainConfig is not None and TorchLinearProbeGeom is not None:
    TrainConfig = TorchTrainConfig  # type: ignore[misc]
    LinearProbeGeom = TorchLinearProbeGeom  # type: ignore[misc]
    build_geometry = torch_build_geometry  # type: ignore[misc]


# -----------------------------------------------------------------------------
# NumPy fallbacks for dataset utilities (mirroring :mod:`dataset`)
# -----------------------------------------------------------------------------

def minmax_scale_np(x: np.ndarray, vmin: float, vmax: float, eps: float = 1e-12) -> np.ndarray:
    """Scale ``x`` to the ``[0, 1]`` interval given global ``vmin``/``vmax``."""

    rng = max(vmax - vmin, eps)
    return (x - vmin) / rng


def pad_or_crop_sinogram_np(sinogram: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pad or crop a 2-D sinogram to ``target_shape`` using zero padding."""

    target_H, target_W = target_shape
    sinogram = np.asarray(sinogram, dtype=np.float64)

    if sinogram.shape[0] > target_H:
        sinogram = sinogram[:target_H, :]
    elif sinogram.shape[0] < target_H:
        pad = target_H - sinogram.shape[0]
        sinogram = np.pad(sinogram, ((0, pad), (0, 0)), mode="constant")

    if sinogram.shape[1] > target_W:
        sinogram = sinogram[:, :target_W]
    elif sinogram.shape[1] < target_W:
        pad = target_W - sinogram.shape[1]
        sinogram = np.pad(sinogram, ((0, 0), (0, pad)), mode="constant")

    return sinogram


def load_hdf5_sample_np(
    input_path: Path,
    target_dir: Optional[Path],
    wavelength: str,
    target_shape: Tuple[int, int],
    sino_min: float,
    sino_max: float,
    img_min: float,
    img_max: float,
    apply_normalization: bool,
    require_target: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pure NumPy implementation of :func:`dataset.load_hdf5_sample`."""

    with h5py.File(str(input_path), "r") as f:
        sinogram = np.array(
            f["simulations"]["time_series_data"][str(wavelength)][()],
            dtype=np.float64,
        )

    sinogram = pad_or_crop_sinogram_np(sinogram, target_shape)
    sinogram = np.flip(sinogram, axis=0)  # match torch.flip(..., dims=[1])

    target = None
    if target_dir is not None:
        target_name = input_path.name.replace(
            ".hdf5", f"_{wavelength}_rec_L1_shearlet.nii"
        )
        target_path = target_dir / target_name
        if target_path.exists():
            try:
                import nibabel as nib  # type: ignore

                nii = nib.load(str(target_path))
                arr = np.asarray(nii.get_fdata(), dtype=np.float32)
                target = arr
            except ImportError:
                if require_target:
                    raise
        elif require_target:
            raise FileNotFoundError(f"Target file not found: {target_path}")

    if apply_normalization:
        sinogram = minmax_scale_np(sinogram, sino_min, sino_max)
        if target is not None:
            target = minmax_scale_np(target, img_min, img_max)

    return sinogram, target


def load_sample(
    input_path: Path,
    target_dir: Optional[Path],
    wavelength: int,
    target_shape: Tuple[int, int],
    sino_min: float,
    sino_max: float,
    img_min: float,
    img_max: float,
    apply_normalization: bool,
    require_target: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a sinogram/target pair using PyTorch dataset helpers when available."""

    if torch_load_hdf5_sample is not None:
        sinogram_t, target_t = torch_load_hdf5_sample(
            input_path=str(input_path),
            target_dir=str(target_dir) if target_dir is not None else "",
            wavelength=str(wavelength),
            target_shape=target_shape,
            sino_min=sino_min,
            sino_max=sino_max,
            img_min=img_min,
            img_max=img_max,
            apply_normalization=apply_normalization,
            require_target=require_target,
        )

        def to_numpy(tensor):
            if tensor is None:
                return None
            arr = tensor
            if hasattr(arr, "detach"):
                arr = arr.detach()
            if hasattr(arr, "cpu"):
                arr = arr.cpu()
            return np.array(arr, dtype=np.float64)

        sinogram = to_numpy(sinogram_t).squeeze(0)
        target = to_numpy(target_t)
        if target is not None and target.ndim >= 3:
            target = target.squeeze()
        return sinogram, target

    return load_hdf5_sample_np(
        input_path=input_path,
        target_dir=target_dir,
        wavelength=str(wavelength),
        target_shape=target_shape,
        sino_min=sino_min,
        sino_max=sino_max,
        img_min=img_min,
        img_max=img_max,
        apply_normalization=apply_normalization,
        require_target=require_target,
    )


# -----------------------------------------------------------------------------
# Numerical helpers (cumtrapz, divergence, PSF construction, kernel)
# -----------------------------------------------------------------------------

def cumulative_trapezoid(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Vectorised cumulative trapezoidal integration matching MATLAB ``cumtrapz``."""

    if y.shape[axis] != x.size:
        raise ValueError("Axis length mismatch between samples and coordinate vector")

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    moved = np.moveaxis(y, axis, 0)
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x must be strictly increasing for trapezoidal integration")

    avg = 0.5 * (moved[1:] + moved[:-1])
    shape = (avg.shape[0],) + (1,) * (avg.ndim - 1)
    contrib = avg * dx.reshape(shape)
    integ = np.cumsum(contrib, axis=0)
    integ = np.concatenate([np.zeros_like(moved[:1]), integ], axis=0)
    return np.moveaxis(integ, 0, axis)


def divergence_2d(Fx: np.ndarray, Fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return ``dFx/dx + dFy/dy`` using centred finite differences."""

    dFx_dx = np.gradient(Fx, dx, axis=1)
    dFy_dy = np.gradient(Fy, dy, axis=0)
    return dFx_dx + dFy_dy


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Generate a 1-D Gaussian PSF of ``size`` taps and standard deviation ``sigma``."""

    if size <= 0:
        raise ValueError("Gaussian PSF size must be positive")
    radius = (size - 1) / 2
    x = np.arange(size, dtype=np.float64) - radius
    psf = np.exp(-(x ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    return psf


def apply_prefilter(sig: np.ndarray, psf: Optional[np.ndarray]) -> np.ndarray:
    """Apply a 1-D convolution along the time axis if ``psf`` is provided."""

    if psf is None:
        return sig
    psf = np.asarray(psf, dtype=np.float64)
    if psf.ndim != 1:
        raise ValueError("Only 1-D PSF filters are supported in this helper")

    out = np.empty_like(sig)
    for det in range(sig.shape[1]):
        for wl in range(sig.shape[2]):
            out[:, det, wl] = np.convolve(sig[:, det, wl], psf, mode="same")
    norm = psf.sum()
    if not np.isclose(norm, 0.0):
        out /= norm
    return out


def bp_kernel(t: np.ndarray) -> np.ndarray:
    """Compute the 2-D kernel used by :func:`backproject_waveeq_linear`.

    The kernel matches the MATLAB implementation::

        K(j,:) = trapz(lambda, H0(j,:,:) .* conj(H0) .* lambda, 3);

    where ``H0`` are Hankel functions of order 0 evaluated at ``lambda * t``.
    """

    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("t must be a 1-D array of time samples")
    if t.size < 2:
        raise ValueError("Need at least two time samples to build the kernel")

    dt = t[-1] - t[0]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time vector must be strictly increasing")

    dlambda = math.pi / dt
    lambda_vals = np.arange(1, t.size + 1, dtype=np.float64) * dlambda
    H = special.hankel1(0, np.outer(t, lambda_vals))

    integrand = H[:, None, :] * np.conjugate(H)[None, :, :] * lambda_vals[None, None, :]
    K_complex = np.trapz(integrand, lambda_vals, axis=2)
    K = -np.imag(K_complex)

    # Enforce symmetry numerically (small imaginary parts can remain due to FP errors)
    K = 0.5 * (K + K.T)
    return K.astype(np.float64)


# -----------------------------------------------------------------------------
# Backprojection core
# -----------------------------------------------------------------------------

def backproject_waveeq_linear(
    sig_mat: np.ndarray,
    fov_x: np.ndarray,
    fov_y: np.ndarray,
    x_position: np.ndarray,
    z_position: np.ndarray,
    speed_of_sound: float,
    sampling_frequency: float,
    start_offset_samples: int,
    psf: Optional[np.ndarray],
    kernel: np.ndarray,
) -> np.ndarray:
    """Port of the MATLAB ``backproject_waveeq_linear`` routine."""

    sig = np.asarray(sig_mat, dtype=np.float64)
    if sig.ndim == 2:
        sig = sig[:, :, None]
    if sig.ndim != 3:
        raise ValueError("Expected sinogram with shape (n_t, n_det, [n_lambda])")

    dt = 1.0 / sampling_frequency
    n_samples, n_det, n_lambda = sig.shape

    acquisition_samples = np.arange(1, n_samples + 1, dtype=np.int64) + start_offset_samples
    acquisition_seconds = acquisition_samples * dt

    sig = cumulative_trapezoid(sig / speed_of_sound, acquisition_seconds, axis=0)
    denom = np.maximum(acquisition_seconds[:, None, None], np.finfo(np.float64).eps)
    sig = sig / denom

    sig = apply_prefilter(sig, psf)

    indices = acquisition_samples - 1
    if kernel.shape[0] <= indices.max() or kernel.shape[1] <= indices.max():
        raise ValueError("Kernel is smaller than the required acquisition indices")
    kernel_sub = kernel[np.ix_(indices, indices)]

    recon = np.zeros((fov_x.shape[0], fov_x.shape[1], n_lambda), dtype=np.float64)

    cos_theta = math.cos(-math.pi / 2)
    sin_theta = math.sin(-math.pi / 2)

    for wl in range(n_lambda):
        accum_x = np.zeros_like(fov_x, dtype=np.float64)
        accum_y = np.zeros_like(fov_x, dtype=np.float64)
        sig_wl = kernel_sub @ sig[:, :, wl] * dt

        for det_idx in range(n_det):
            distance = np.sqrt(
                (fov_x - x_position[det_idx]) ** 2 + (fov_y - z_position[det_idx]) ** 2
            )
            arrival = (
                (distance / speed_of_sound - acquisition_seconds[0]) * sampling_frequency + 1.0
            )
            time_indices = np.rint(arrival).astype(np.int64)
            valid = (time_indices >= 1) & (time_indices <= n_samples)

            contrib = np.zeros_like(fov_x, dtype=np.float64)
            valid_indices = time_indices[valid] - 1
            if valid_indices.size > 0:
                contrib[valid] = sig_wl[valid_indices, det_idx]

            accum_x += contrib * cos_theta * dt * speed_of_sound
            accum_y += contrib * sin_theta * dt * speed_of_sound

        div = divergence_2d(accum_x, accum_y, dx=fov_x[0, 1] - fov_x[0, 0], dy=fov_y[1, 0] - fov_y[0, 0])
        recon[:, :, wl] = np.fliplr(div)

    return recon.squeeze()


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def build_field_of_view(geom: LinearProbeGeom) -> Tuple[np.ndarray, np.ndarray]:
    """Return the spatial sampling grid in meters."""

    xs = geom.x0_m + np.arange(geom.nx, dtype=np.float64) * geom.dx_m
    ys = geom.y0_m + np.arange(geom.ny, dtype=np.float64) * geom.dy_m
    return np.meshgrid(xs, ys)


GEOMETRY_FIELDS = {
    "n_det": int,
    "pitch_m": float,
    "t0_s": float,
    "dt_s": float,
    "n_t": int,
    "c_m_s": float,
    "x0_m": float,
    "y0_m": float,
    "dx_m": float,
    "dy_m": float,
    "nx": int,
    "ny": int,
    "array_x0_m": float,
    "array_y_m": float,
}


def parse_geometry_overrides(parser: argparse.ArgumentParser) -> None:
    """Add geometry override options to ``parser`` for convenience."""

    for field, typ in GEOMETRY_FIELDS.items():
        parser.add_argument(
            f"--{field}",
            type=typ,
            help=f"Override TrainConfig.{field}",
        )


def apply_geometry_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Return a copy of ``cfg`` with CLI overrides applied."""

    cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    for field, typ in GEOMETRY_FIELDS.items():
        value = getattr(args, field, None)
        if value is not None:
            cfg_dict[field] = typ(value)
    return cfg.__class__(**cfg_dict)


def load_or_build_kernel(
    n_samples: int,
    dt: float,
    start_offset: int,
    cache_path: Optional[Path],
) -> np.ndarray:
    """Load a cached kernel if available, otherwise build it on the fly."""

    if cache_path is not None and cache_path.exists():
        kernel = np.load(str(cache_path))
        return kernel

    total_samples = n_samples + start_offset
    t = (np.arange(1, total_samples + 1, dtype=np.float64)) * dt
    kernel = bp_kernel(t)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), kernel)

    return kernel


def load_psf_from_args(args: argparse.Namespace) -> Optional[np.ndarray]:
    """Instantiate the PSF filter according to CLI arguments."""

    if args.psf_path:
        psf_path = Path(args.psf_path)
        if not psf_path.exists():
            raise FileNotFoundError(f"PSF file not found: {psf_path}")
        if psf_path.suffix == ".npy":
            psf = np.load(str(psf_path))
        else:
            psf = np.loadtxt(str(psf_path))
        return np.asarray(psf, dtype=np.float64)

    if args.psf_size and args.psf_sigma:
        return gaussian_psf(args.psf_size, args.psf_sigma)

    return None


def save_outputs(
    output_path: Path,
    reconstruction: np.ndarray,
    target: Optional[np.ndarray],
    png_path: Optional[Path],
) -> None:
    """Persist the reconstruction (and optionally the target) to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), reconstruction.astype(np.float32))

    if png_path is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(6, 6))
            if reconstruction.ndim == 3:
                img = reconstruction[..., 0]
            else:
                img = reconstruction
            plt.imshow(img, cmap="gray")
            plt.title("Backprojection")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(str(png_path), dpi=150)
            plt.close()
        except ImportError:
            raise RuntimeError("matplotlib is required to export PNG figures")

    if target is not None:
        target_path = output_path.with_suffix(".target.npy")
        np.save(str(target_path), target.astype(np.float32))


def summarise_result(reconstruction: np.ndarray) -> str:
    """Return a JSON summary with shape and intensity statistics."""

    stats = {
        "shape": reconstruction.shape,
        "min": float(np.nanmin(reconstruction)),
        "max": float(np.nanmax(reconstruction)),
        "mean": float(np.nanmean(reconstruction)),
        "std": float(np.nanstd(reconstruction)),
    }
    return json.dumps(stats, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to the input .hdf5 sinogram")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional dataset root used to locate targets")
    parser.add_argument("--sino-dir", type=Path, default=None, help="Relative path containing sinograms (under data-root)")
    parser.add_argument("--recs-dir", type=Path, default=None, help="Relative path containing reference reconstructions")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Dataset split folder (if using data-root paths)")
    parser.add_argument("--wavelength", type=int, default=None, help="Acquisition wavelength to select (defaults to TrainConfig.wavelength)")
    parser.add_argument("--output", type=Path, default=None, help="Output .npy path (defaults to input stem + '_bp.npy')")
    parser.add_argument("--png", type=Path, default=None, help="Optional PNG file for quick visual inspection")
    parser.add_argument("--start-offset", type=int, default=0, help="Number of missing samples at the beginning of the sinogram")
    parser.add_argument("--kernel-cache", type=Path, default=None, help="Optional .npy path to cache/reuse the backprojection kernel")
    parser.add_argument("--psf-path", type=str, default=None, help="Path to a custom PSF (np.load compatible)")
    parser.add_argument("--psf-size", type=int, default=None, help="Size of a Gaussian PSF (requires --psf-sigma)")
    parser.add_argument("--psf-sigma", type=float, default=None, help="Sigma for the Gaussian PSF (requires --psf-size)")
    parser.add_argument("--no-normalization", action="store_true", help="Disable [0,1] min-max normalisation applied by the dataset")
    parser.add_argument("--require-target", action="store_true", help="Raise an error if the target NIfTI is missing")
    parser.add_argument("--summary", action="store_true", help="Print a JSON summary of the reconstruction")

    parse_geometry_overrides(parser)
    return parser


def resolve_input_path(args: argparse.Namespace) -> Path:
    """Resolve the absolute path to the input HDF5 file."""

    if args.data_root and args.sino_dir:
        base = args.data_root / args.sino_dir
        if args.split:
            base = base / args.split
        candidate = base / args.input
        if candidate.exists():
            return candidate
    return args.input


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    cfg = apply_geometry_overrides(cfg, args)
    wavelength = args.wavelength if args.wavelength is not None else getattr(cfg, "wavelength", 800)

    geom = build_geometry(cfg)
    fov_x, fov_y = build_field_of_view(geom)

    input_path = resolve_input_path(args)
    if not input_path.exists():
        raise FileNotFoundError(f"Input sinogram not found: {input_path}")

    target_dir = None
    if args.data_root and args.recs_dir:
        target_dir = args.data_root / args.recs_dir
    elif args.recs_dir:
        target_dir = args.recs_dir
    if target_dir is not None and not target_dir.exists():
        if args.require_target:
            raise FileNotFoundError(f"Target directory not found: {target_dir}")
        target_dir = None

    sinogram, target = load_sample(
        input_path=input_path,
        target_dir=target_dir,
        wavelength=wavelength,
        target_shape=(geom.n_det, geom.n_t),
        sino_min=getattr(cfg, "sino_min", -1.0),
        sino_max=getattr(cfg, "sino_max", 1.0),
        img_min=getattr(cfg, "img_min", 0.0),
        img_max=getattr(cfg, "img_max", 1.0),
        apply_normalization=not args.no_normalization,
        require_target=args.require_target,
    )

    # Dataset sinograms are (n_det, n_t); transpose to (n_t, n_det)
    if sinogram.ndim == 2:
        sinogram = sinogram.T
    elif sinogram.ndim == 3:
        sinogram = np.transpose(sinogram, (2, 1, 0))
    else:
        raise ValueError(f"Unexpected sinogram shape: {sinogram.shape}")

    kernel = load_or_build_kernel(
        n_samples=sinogram.shape[0],
        dt=geom.dt_s,
        start_offset=args.start_offset,
        cache_path=args.kernel_cache,
    )

    psf = load_psf_from_args(args)

    reconstruction = backproject_waveeq_linear(
        sig_mat=sinogram,
        fov_x=fov_x,
        fov_y=fov_y,
        x_position=geom.det_x.astype(np.float64),
        z_position=geom.det_y.astype(np.float64),
        speed_of_sound=geom.c_m_s,
        sampling_frequency=1.0 / geom.dt_s,
        start_offset_samples=args.start_offset,
        psf=psf,
        kernel=kernel,
    )

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix("")
        output_path = output_path.with_name(output_path.name + "_bp.npy")
    png_path = args.png

    save_outputs(Path(output_path), reconstruction, target, Path(png_path) if png_path else None)

    if args.summary:
        print(summarise_result(reconstruction))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
