import os
from typing import Tuple

import h5py
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

nib.imageglobals.logger.setLevel(40)


def minmax_scale(x: torch.Tensor, vmin: float, vmax: float, eps: float = 1e-12) -> torch.Tensor:
    """Scale to [0,1] with global min/max; safely handle zero-range."""
    rng = max(vmax - vmin, eps)
    return (x - vmin) / rng


def pad_or_crop_sinogram(sinogram: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Pad/crop a 2D sinogram to the desired detector/time shape."""
    target_H, target_W = target_shape
    H, W = sinogram.shape

    if W > target_W:
        sinogram = sinogram[:, :target_W]
    elif W < target_W:
        pad_width = target_W - W
        sinogram = F.pad(sinogram, (0, pad_width), mode='constant', value=0)

    if H > target_H:
        sinogram = sinogram[:target_H, :]
    elif H < target_H:
        pad_height = target_H - H
        sinogram = F.pad(sinogram, (0, 0, 0, pad_height), mode='constant', value=0)

    return sinogram


def load_hdf5_sample(
    input_path: str,
    target_dir: str,
    wavelength: str,
    target_shape: Tuple[int, int],
    sino_min: float,
    sino_max: float,
    img_min: float,
    img_max: float,
    apply_normalization: bool = True,
    require_target: bool = True,
):
    """Load a single sinogram/target pair applying the dataset preprocessing pipeline."""
    with h5py.File(input_path, 'r') as f:
        sinogram = torch.tensor(
            f['simulations']['time_series_data'][str(wavelength)][()],
            dtype=torch.float32
        )

    sinogram = pad_or_crop_sinogram(sinogram, target_shape).unsqueeze(0)

    fname = os.path.basename(input_path).replace(
        ".hdf5", f"_{wavelength}_rec_L1_shearlet.nii"
    )
    target_path = os.path.join(target_dir, fname)
    target = None
    if os.path.exists(target_path):
        nifti = nib.load(target_path)
        target_np = nifti.get_fdata().astype('float32')
        if target_np.ndim == 2:
            target = torch.tensor(target_np).unsqueeze(0)
        else:
            target = torch.tensor(target_np)
    elif require_target:
        raise FileNotFoundError(f"Target file not found: {target_path}")

    sinogram = torch.flip(sinogram, dims=[1])

    if apply_normalization:
        sinogram = minmax_scale(sinogram, sino_min, sino_max)
        if target is not None:
            target = minmax_scale(target, img_min, img_max)
            # plt.imshow(target[0, :, :], cmap='gray')
            # plt.show()

    return sinogram, target


class HDF5Dataset(Dataset):
    def __init__(self, input_dir, target_dir, sino_min: float, sino_max: float, img_min: float, img_max: float, split='train', wavelength=800, target_shape=(128, 1640)):
        super().__init__()
        self.input_dir = input_dir + "/" + split
        self.target_dir = target_dir
        self.smin, self.smax = float(sino_min), float(sino_max)
        self.imin, self.imax = float(img_min), float(img_max)
        self.wavelength = str(wavelength)  # es. '750'
        self.file_list = sorted([
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.endswith('.hdf5')
        ])
        self.target_shape = target_shape

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_path = self.file_list[idx]
        sinogram, target = load_hdf5_sample(
            input_path=input_path,
            target_dir=self.target_dir,
            wavelength=self.wavelength,
            target_shape=self.target_shape,
            sino_min=self.smin,
            sino_max=self.smax,
            img_min=self.imin,
            img_max=self.imax,
            apply_normalization=True,
            require_target=True,
        )

        return sinogram, target
