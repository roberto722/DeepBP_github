import os
import h5py
import nibabel as nib
import numpy as np

def compute_stats(sinogram_dir, groundtruth_dir, hdf5_key="simulations/time_series_data/800", nii_suffix="_800_rec_L1_shearlet.nii"):
    # Sinogram stats
    s_min, s_max = float('inf'), float('-inf')
    s_sum, s_sum2, s_count = 0.0, 0.0, 0

    # Target stats
    t_min, t_max = float('inf'), float('-inf')
    t_sum, t_sum2, t_count = 0.0, 0.0, 0

    files = sorted([
        f for f in os.listdir(sinogram_dir) if f.endswith('.hdf5')
    ])

    for fname in files:
        # === SINOGRAM ===
        hdf5_path = os.path.join(sinogram_dir, fname)
        with h5py.File(hdf5_path, 'r') as f:
            sinogram = f[hdf5_key][()].astype(np.float32)

        s_min = min(s_min, sinogram.min())
        s_max = max(s_max, sinogram.max())
        s_sum += sinogram.sum()
        s_sum2 += np.square(sinogram).sum()
        s_count += sinogram.size

        # === TARGET ===
        target_name = fname.replace('.hdf5', nii_suffix)
        nii_path = os.path.join(groundtruth_dir, target_name)

        if os.path.exists(nii_path):
            image = nib.load(nii_path).get_fdata().astype(np.float32)

            t_min = min(t_min, image.min())
            t_max = max(t_max, image.max())
            t_sum += image.sum()
            t_sum2 += np.square(image).sum()
            t_count += image.size
        else:
            print(f"Attenzione: file target mancante → {nii_path}")

    # === Final statistics ===
    s_mean = s_sum / s_count
    s_std = np.sqrt(s_sum2 / s_count - s_mean**2)

    t_mean = t_sum / t_count
    t_std = np.sqrt(t_sum2 / t_count - t_mean**2)

    return {
        "sinogram_min": s_min,
        "sinogram_max": s_max,
        "sinogram_mean": s_mean,
        "sinogram_std": s_std,
        "target_min": t_min,
        "target_max": t_max,
        "target_mean": t_mean,
        "target_std": t_std
    }


if __name__ == "__main__":
    results = compute_stats(
        sinogram_dir="E:\Scardigno\datasets_transformer_proj\Forearm2000_hdf5\\train_val_tst\\train",
        groundtruth_dir="E:\Scardigno\datasets_transformer_proj/Forearm2000_recs/L1_Shearlet",
        hdf5_key="simulations/time_series_data/800",
        nii_suffix="_800_rec_L1_shearlet.nii"
    )

    print("📊 SINOGRAM")
    print(f"min:  {results['sinogram_min']:.4f}")
    print(f"max:  {results['sinogram_max']:.4f}")
    print(f"mean: {results['sinogram_mean']:.4f}")
    print(f"std:  {results['sinogram_std']:.4f}")

    print("\n📊 TARGET")
    print(f"min:  {results['target_min']:.4f}")
    print(f"max:  {results['target_max']:.4f}")
    print(f"mean: {results['target_mean']:.4f}")
    print(f"std:  {results['target_std']:.4f}")
