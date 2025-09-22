import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ====== Backprojection Torch (ring OR linear) ======
class BackProjectionReconstructorTorch:
    """
    Due modalità:
      - geometry="ring": geometria circolare (come prima: campo vettoriale + divergenza)
      - geometry="linear": array lineare (derivata temporale + delay-and-sum + peso 1/r opzionale)
    """

    def __init__(self,
                 speed_of_sound: float,
                 sampling_frequency: float,
                 # --- parametri ring (ignorati in linear) ---
                 angular_coverage_deg: float = 180.0,
                 detector_radius: float = 0.02,
                 # --- parametri linear ---
                 geometry: str = "linear",
                 pitch: Optional[float] = None,     # distanza tra elementi (m)
                 y0: float = 0.0,                    # quota della sonda (m), es. 0.0
                 derivative_order: int = 1,         # 1 o 2 (derivata temporale)
                 use_inv_r_weight: bool = True,     # usa peso 1/r
                 cropped_or_unrecorded_at_start: int = 0,
                 psf_1d: Optional[torch.Tensor] = None,   # 1D kernel lungo il tempo
                 K_torch: Optional[torch.Tensor] = None,  # (Nt x Nt) solo per ring/opzionale
                 device: Optional[torch.device] = None):

        self.c = float(speed_of_sound)
        self.fs = float(sampling_frequency)
        self.angular_coverage = float(angular_coverage_deg)
        self.r = float(detector_radius)
        self.start_crop = int(cropped_or_unrecorded_at_start)

        self.geometry = geometry.lower()
        assert self.geometry in ("ring", "linear")
        self.pitch = pitch
        self.y0 = float(y0)
        self.derivative_order = int(derivative_order)
        assert self.derivative_order in (1, 2)
        self.use_inv_r_weight = bool(use_inv_r_weight)

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # PSF: salva come [1, 1, K] per conv1d con padding='same'
        if psf_1d is None:
            self.psf = torch.tensor([1.0], dtype=torch.float32)
        else:
            self.psf = psf_1d.detach().float().cpu()
        self.psf = self.psf.view(1, 1, -1).to(self.device)
        self.psf_norm = float(self.psf.sum().item()) if self.psf.sum().abs() > 0 else 1.0

        # Kernel K opzionale (usato solo nel ramo ring)
        self.K = None
        if K_torch is not None:
            self.K = K_torch.detach().float().to(self.device)

    @staticmethod
    def _cumtrapz_constant_dt(x: torch.Tensor, dt: float, dim: int = 0) -> torch.Tensor:
        x1 = torch.narrow(x, dim, 1, x.size(dim) - 1)
        x0 = torch.narrow(x, dim, 0, x.size(dim) - 1)
        mid = 0.5 * (x1 + x0) * dt
        y0 = torch.zeros_like(
            torch.index_select(x, dim, torch.tensor([0], device=x.device, dtype=torch.long))
        )
        y = torch.cat([y0, torch.cumsum(mid, dim=dim)], dim=dim)
        return y

    def _prefilter_time_conv(self, s: torch.Tensor) -> torch.Tensor:
        Nt, Np = s.shape
        x = s.t().unsqueeze(0)  # [1, Np, Nt]
        k = self.psf.repeat(Np, 1, 1)  # [Np, 1, K]
        pad = (k.shape[-1] - 1) // 2
        y = F.conv1d(x, k, padding=pad, groups=Np)  # [1, Np, Nt]
        y = y / self.psf_norm
        return y.squeeze(0).t()  # [Nt, Np]

    def _time_derivative(self, s: torch.Tensor, order: int) -> torch.Tensor:
        """Derivata temporale discreta (ordine 1 o 2) depthwise lungo il tempo."""
        Nt, Np = s.shape
        x = s.t().unsqueeze(0)  # [1, Np, Nt]
        if order == 1:
            ker = torch.tensor([-0.5, 0.0, 0.5], device=s.device, dtype=torch.float32).view(1, 1, 3)
        else:  # order == 2
            ker = torch.tensor([1.0, -2.0, 1.0], device=s.device, dtype=torch.float32).view(1, 1, 3)
        k = ker.repeat(Np, 1, 1)  # [Np,1,K]
        y = F.conv1d(x, k, padding=1, groups=Np).squeeze(0).t()  # [Nt, Np]
        # scala per dt^order (derivata continua ~ differenze finite / dt^order)
        dt = 1.0 / self.fs
        y = y / (dt ** order)
        return y

    def reconstruct_single(self,
                           sinogram: torch.Tensor,   # [1, Nt, Np] oppure [Nt, Np]
                           fov_x: torch.Tensor,      # [H, W]
                           fov_y: torch.Tensor       # [H, W]
                           ) -> torch.Tensor:
        device = self.device

        # Prepara sinogramma
        if sinogram.ndim == 3:
            sig = sinogram.squeeze(0).to(device).float()  # [Nt, Np]
        else:
            sig = sinogram.to(device).float()
        Nt, Np = sig.shape
        dt = 1.0 / self.fs

        # Tempi
        t_samples = torch.arange(1, Nt + 1, device=device, dtype=torch.float32) + self.start_crop
        t_sec = t_samples * dt  # [Nt]

        # Pre-elaborazione comune: integrazione radiale + PSF
        sig_scaled = sig / self.c
        sig_int = self._cumtrapz_constant_dt(sig_scaled, dt, dim=0)  # [Nt, Np]
        sig_radial = sig_int / t_sec.view(-1, 1)
        sig_pref = self._prefilter_time_conv(sig_radial)  # [Nt, Np]

        # Precompute pixel grid
        fov_x = fov_x.to(device).float()
        fov_y = fov_y.to(device).float()
        H, W = fov_x.shape
        X = fov_x.reshape(-1)[:, None]  # [HW, 1]
        Y = fov_y.reshape(-1)[:, None]  # [HW, 1]

        if self.geometry == "ring":
            # ===== Geometria circolare (come prima) =====
            angular_offset = (180.0 - self.angular_coverage) / 2.0
            th = -torch.linspace(angular_offset, 180.0 - angular_offset, Np, device=device) * (np.pi / 180.0)
            cos_th, sin_th = torch.cos(th), torch.sin(th)

            # Kernel K opzionale
            if self.K is not None:
                if self.K.shape != (Nt, Nt):
                    raise ValueError(f"Dimensione K {self.K.shape} non combacia con Nt={Nt}")
                sig_filt = (self.K @ sig_pref) * dt
            else:
                sig_filt = sig_pref

            tx = self.r * cos_th  # [Np]
            ty = self.r * sin_th  # [Np]

            dx = X - tx.view(1, -1)
            dy = Y - ty.view(1, -1)
            dist = torch.sqrt(dx * dx + dy * dy)  # [HW, Np]
            tof = dist / self.c

            T_float = (tof - t_sec[0]) * self.fs + 1.0
            T_idx = torch.round(T_float).long() - 1
            T_idx = torch.clamp(T_idx, 0, Nt - 1)

            p_ids = torch.arange(Np, device=device).view(1, -1).expand(T_idx.shape[0], -1)
            vals = sig_filt[T_idx, p_ids]  # [HW, Np]

            weight = dt * self.c
            contrib_x = (vals * cos_th.view(1, -1)).sum(dim=1) * weight
            contrib_y = (vals * sin_th.view(1, -1)).sum(dim=1) * weight

            p0x = contrib_x.view(H, W)
            p0y = contrib_y.view(H, W)

            # Divergenza + fliplr
            div = divergence_on_grid_torch(p0x, p0y, fov_x, fov_y)
            p0 = torch.flip(div, dims=[1])
            return p0  # [H, W]

        else:
            # ===== Geometria lineare =====
            if self.pitch is None:
                raise ValueError("Per geometry='linear' devi specificare pitch (m).")
            aperture = self.pitch * (Np - 1)
            tx = torch.linspace(-aperture / 2, aperture / 2, Np, device=device)
            ty = torch.full((Np,), self.y0, device=device)

            # Derivata temporale (ordine 1 o 2)
            sig_dt = self._time_derivative(sig_pref, order=self.derivative_order)  # [Nt, Np]

            dx = X - tx.view(1, -1)
            dy = Y - ty.view(1, -1)
            dist = torch.sqrt(dx * dx + dy * dy)  # [HW, Np]
            tof = dist / self.c

            T_float = (tof - t_sec[0]) * self.fs + 1.0
            T_idx = torch.round(T_float).long() - 1
            T_idx = torch.clamp(T_idx, 0, Nt - 1)

            p_ids = torch.arange(Np, device=device).view(1, -1).expand(T_idx.shape[0], -1)
            vals = sig_dt[T_idx, p_ids]  # [HW, Np]

            if self.use_inv_r_weight:
                eps = 1e-9
                weights = 1.0 / (dist + eps)
                vals = vals * weights

            # Delay-and-sum (scalare). Molti UBP usano solo la somma; dt è già considerato nella derivata.
            p0 = vals.sum(dim=1).view(H, W)
            # opzionale: normalizzazione per Np
            # p0 = p0 / Np
            return p0

# ====== (OPZIONALE) Kernel K come in MATLAB: bp_kernel.m ======
def build_bp_kernel_np(t_sec: np.ndarray) -> np.ndarray:
    """
    Replica del bp_kernel(t) MATLAB usando SciPy (hankel1).
    Restituisce K (Nt x Nt), poi potrai convertirlo in torch.
    Se SciPy non è disponibile, solleva ImportError.
    """
    try:
        from scipy.special import hankel1
    except ImportError as e:
        raise ImportError("Per build_bp_kernel_np serve SciPy (scipy.special.hankel1). "
                          "Installa scipy oppure usa K=None.") from e

    t = np.asarray(t_sec).ravel()
    Nt = t.size
    L = t[-1] - t[0]
    if L <= 0:
        raise ValueError("t deve essere strettamente crescente.")
    dlambda = np.pi / L
    lam = np.arange(1, Nt + 1, dtype=float) * dlambda  # (Nt,)

    # H0(j, k) = H_0^(1)( t[j] * lam[k] )
    H0 = hankel1(0, np.outer(t, lam))  # (Nt, Nt)

    integrand = (H0 * np.conj(H0)) * lam  # |H0|^2 * lambda
    # integrazione su lam (asse=1), ripetuta per ogni riga j
    K_row = np.trapz(integrand, x=lam, axis=1)  # (Nt,)
    K = -np.imag(K_row)[:, None].repeat(Nt, axis=1)  # come nel MATLAB: righe uguali
    return K.astype(np.float32)


# ====== Utility: divergence 2D su griglia regolare ======
def divergence_on_grid_torch(fx: torch.Tensor, fy: torch.Tensor,
                             fov_x: torch.Tensor, fov_y: torch.Tensor) -> torch.Tensor:
    """
    Divergenza di (fx, fy) su griglia (fov_x, fov_y).
    fx, fy, fov_x, fov_y: [H, W] (torch)
    """
    device = fx.device
    H, W = fx.shape
    dx = (fov_x[0, 1] - fov_x[0, 0]) if W > 1 else torch.tensor(1.0, device=device)
    dy = (fov_y[1, 0] - fov_y[0, 0]) if H > 1 else torch.tensor(1.0, device=device)

    # derivate centrali con padding ai bordi (forward/backward)
    # dfx/dx
    dfx_dx = torch.zeros_like(fx)
    dfx_dx[:, 1:-1] = (fx[:, 2:] - fx[:, :-2]) / (2 * dx)
    dfx_dx[:, 0]    = (fx[:, 1] - fx[:, 0]) / dx
    dfx_dx[:, -1]   = (fx[:, -1] - fx[:, -2]) / dx

    # dfy/dy
    dfy_dy = torch.zeros_like(fy)
    dfy_dy[1:-1, :] = (fy[2:, :] - fy[:-2, :]) / (2 * dy)
    dfy_dy[0, :]    = (fy[1, :] - fy[0, :]) / dy
    dfy_dy[-1, :]   = (fy[-1, :] - fy[-2, :]) / dy

    return dfx_dx + dfy_dy

# ==========================
# ==== INTEGRAZIONE CON IL TUO DATASET ====
# (Usa le tue definizioni HDF5Dataset, load_hdf5_sample, ecc.)
# ==========================
# (Incolla qui le tue definizioni già fornite)
import h5py
import nibabel as nib
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
nib.imageglobals.logger.setLevel(40)

def minmax_scale(x: torch.Tensor, vmin: float, vmax: float, eps: float = 1e-12) -> torch.Tensor:
    rng = max(vmax - vmin, eps)
    return (x - vmin) / rng

def pad_or_crop_sinogram(sinogram: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
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

    # flip temporale come nel tuo codice
    sinogram = torch.flip(sinogram, dims=[1])

    if apply_normalization:
        sinogram = minmax_scale(sinogram, sino_min, sino_max)
        if target is not None:
            target = minmax_scale(target, img_min, img_max)

    return sinogram, target

class HDF5Dataset(Dataset):
    def __init__(self, input_dir, target_dir, sino_min: float, sino_max: float, img_min: float, img_max: float,
                 split='train', wavelength=800, target_shape=(128, 1640)):
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
        return sinogram, target, input_path


# ==========================
# ====== ESEMPIO D’USO =====
# ==========================
if __name__ == "__main__":
    # ---- Config ----
    INPUT_DIR = "E:/Scardigno/datasets_transformer_proj/Forearm2000_hdf5/train_val_tst"  # <-- set
    TARGET_DIR = "E:/Scardigno/datasets_transformer_proj/Forearm2000_recs/L1_Shearlet"  # <-- set
    SPLIT       = "tst"                    # 'train' | 'val' | 'tst'
    WL          = 800
    TARGET_SHAPE = (128, 1640)             # (Nt, Np) per sinogrammi
    # Valori di normalizzazione (adatta ai tuoi range)
    SINO_MIN, SINO_MAX = -1.0, 1.0
    IMG_MIN, IMG_MAX   = 0.0, 1.0

    # Parametri di backprojection
    c   = 1540.0
    fs  = 31.25e6
    start_crop = 0
    # Geometria lineare
    pitch_m = 0.0003  # 0.3 mm tra elementi (adatta ai tuoi dati)
    y0_m = 0.0  # linea trasduttori su y=0 (adatta se necessario)


    # Griglia immagine (FOV): es. 128x128 in metri
    H, W = 128, 128
    x = torch.linspace(-0.01982, 0.01982, W)  # 20 mm
    y = torch.linspace(-0.01982, 0.01982, H)
    fov_x, fov_y = torch.meshgrid(y*0 + x, y, indexing='xy')  # fov_x: [H,W], fov_y: [H,W]
    # Nota: sopra ho costruito (x,y) corretti per indexing='xy'

    # PSF (se non ne hai uno reale): identità
    psf_1d = torch.tensor([1.0], dtype=torch.float32)

    # (Opzionale) Costruisci K(t) come MATLAB
    Nt = TARGET_SHAPE[0]
    t_samples_np = (np.arange(1, Nt + 1) + start_crop) / fs
    try:
        K_np = build_bp_kernel_np(t_samples_np)  # (Nt, Nt)
        K_torch = torch.from_numpy(K_np)
    except ImportError:
        print("SciPy non disponibile: procedo con K=None (baseline).")
        K_torch = None

    # Reconstructor
    recon = BackProjectionReconstructorTorch(
        speed_of_sound=c,
        sampling_frequency=fs,
        geometry="linear",
        pitch=pitch_m,
        y0=y0_m,
        derivative_order=1,  # 1 o 2
        use_inv_r_weight=True,
        # i seguenti ignorati in linear:
        angular_coverage_deg=180.0,
        detector_radius=0.02,
        cropped_or_unrecorded_at_start=start_crop,
        psf_1d=psf_1d,
        K_torch=K_torch,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Dataset & Loader
    ds = HDF5Dataset(
        input_dir=INPUT_DIR,
        target_dir=TARGET_DIR,
        sino_min=SINO_MIN, sino_max=SINO_MAX,
        img_min=IMG_MIN, img_max=IMG_MAX,
        split=SPLIT, wavelength=WL,
        target_shape=TARGET_SHAPE
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Output dirs
    out_dir_png = os.path.join(TARGET_DIR, f"bp_{WL}", "png")
    out_dir_nii = os.path.join(TARGET_DIR, f"bp_{WL}", "nii")
    os.makedirs(out_dir_png, exist_ok=True)
    os.makedirs(out_dir_nii, exist_ok=True)

    # Loop
    for sinogram, target, input_path in dl:
        # sinogram: [1, Nt, Np]
        sinogram = sinogram.to(recon.device)
        p0 = recon.reconstruct_single(sinogram[0], fov_x, fov_y)  # [H, W]
        p0_cpu = p0.detach().cpu()

        # Salvataggio NIfTI accanto al target (o nella cartella dedicata)
        base = os.path.basename(input_path[0]).replace(".hdf5", f"_{WL}_bp.nii.gz")
        nii_path = os.path.join(out_dir_nii, base)

        affine = np.eye(4, dtype=np.float32)
        nib.save(nib.Nifti1Image(p0_cpu.numpy().astype(np.float32), affine), nii_path)

        # PNG
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(p0_cpu.numpy(), cmap='gray')
        plt.axis('off')
        png_path = os.path.join(out_dir_png, base.replace(".nii.gz", ".png"))
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Salvato: {nii_path} | {png_path}")
