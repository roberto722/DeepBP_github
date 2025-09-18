import os
import h5py
import torch
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

    def _pad_or_crop(self, sinogram):
        target_H, target_W = self.target_shape
        H, W = sinogram.shape

        if W > target_W:
            sinogram = sinogram[:, :target_W]
        elif W < target_W:
            pad_width = target_W - W
            sinogram = torch.nn.functional.pad(sinogram, (0, pad_width), mode='constant', value=0)

        if H > target_H:
            sinogram = sinogram[:target_H, :]
        elif H < target_H:
            pad_height = target_H - H
            sinogram = torch.nn.functional.pad(sinogram, (0, 0, 0, pad_height), mode='constant', value=0)

        return sinogram

    def __getitem__(self, idx):
        input_path = self.file_list[idx]

        # === Leggi sinogramma HDF5 ===
        with h5py.File(input_path, 'r') as f:
            sinogram = torch.tensor(
                f['simulations']['time_series_data'][self.wavelength][()],
                dtype=torch.float32
            )
        sinogram = self._pad_or_crop(sinogram).unsqueeze(0)  # (1, 128, 1640)

        # === Costruisci path per il target .nii ===
        fname = os.path.basename(input_path).replace(".hdf5", f"_{self.wavelength}_rec_L1_shearlet.nii")
        target_path = os.path.join(self.target_dir, fname)

        # === Leggi target .nii ===
        nifti = nib.load(target_path)
        target_np = nifti.get_fdata().astype('float32')
        if target_np.ndim == 2:
            target = torch.tensor(target_np).unsqueeze(0)
        else:
            target = torch.tensor(target_np)

        # Flip verticale (asse 0: detector elements) o orizzontale (asse 1: time samples)
        sinogram = torch.flip(sinogram, dims=[1]) # flip detector axis

        sinogram = minmax_scale(sinogram, self.smin, self.smax)
        target = minmax_scale(target,  self.imin, self.imax)

        return sinogram, target
