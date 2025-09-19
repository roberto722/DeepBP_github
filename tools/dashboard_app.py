"""Streamlit dashboard per esplorare il modello BPTransformer."""
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import load_hdf5_sample, minmax_scale
from main import (
    TrainConfig,
    create_model,
    load_checkpoint,
    run_inference_steps,
)


def list_hdf5_files(base_dir: str) -> List[str]:
    """Return sorted list of .hdf5 files in a directory."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(f for f in os.listdir(base_dir) if f.endswith(".hdf5"))


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().squeeze().numpy()


def plot_outputs(images: List[Tuple[str, np.ndarray]], sinogram: bool = False):
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, images):
        cmap = "magma" if "Sinogramma" in title else "gray"
        ax.imshow(data, cmap=cmap, aspect="auto" if sinogram and "Sinogramma" in title else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="BPTransformer Dashboard", layout="wide")
    st.title("Dashboard BPTransformer")

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"Dispositivo: {device}")

    work_dir = cfg.work_dir
    ckpts = [name for name in ("best.pt", "last.pt") if os.path.exists(os.path.join(work_dir, name))]

    model = create_model(cfg, device)
    selected_ckpt: Optional[str] = None
    if ckpts:
        selected_ckpt = st.sidebar.selectbox("Checkpoint", ckpts, index=0)
        if selected_ckpt:
            try:
                load_checkpoint(model, cfg, selected_ckpt, map_location=device)
                st.sidebar.success(f"Caricato {selected_ckpt}")
            except FileNotFoundError as exc:
                st.sidebar.error(str(exc))
    else:
        st.sidebar.warning(
            "Nessun checkpoint trovato. Verranno usati pesi random inizializzati."
        )

    input_root = os.path.join(cfg.data_root, cfg.sino_dir)
    target_root = os.path.join(cfg.data_root, cfg.recs_dir)
    splits = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))] if os.path.isdir(input_root) else []

    st.header("Selezione del file")
    selected_split = None
    selected_file = None
    if splits:
        selected_split = st.selectbox("Split disponibile", splits)
        split_dir = os.path.join(input_root, selected_split)
        files = list_hdf5_files(split_dir)
        if files:
            selected_file = st.selectbox("File dal dataset", files)
        else:
            st.info("Nessun file .hdf5 trovato nello split selezionato.")
    else:
        st.info("Cartella dei sinogrammi non trovata o vuota: controlla TrainConfig.data_root.")

    uploaded_file = st.file_uploader("Oppure carica un file .hdf5", type=["hdf5"])

    sample_path = None
    sample_label = None
    temp_path = None
    if selected_file and selected_split:
        sample_path = os.path.join(input_root, selected_split, selected_file)
        sample_label = f"{selected_split}/{selected_file}"
    elif uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        sample_path = temp_path
        sample_label = uploaded_file.name

    if sample_path:
        st.subheader("Risultati")
        try:
            sinogram_raw, target = load_hdf5_sample(
                input_path=sample_path,
                target_dir=target_root,
                wavelength=str(cfg.wavelength),
                target_shape=(cfg.n_det, cfg.n_t),
                sino_min=cfg.sino_min,
                sino_max=cfg.sino_max,
                img_min=cfg.img_min,
                img_max=cfg.img_max,
                apply_normalization=False,
                require_target=False,
            )

            sino_norm, bp_img, pred_img, iter_imgs = run_inference_steps(
                model,
                sinogram_raw,
                cfg,
                device=device,
                normalize=True,
            )

            sino_plot = tensor_to_numpy(sino_norm[0])
            bp_plot = tensor_to_numpy(bp_img[0])
            pred_plot = tensor_to_numpy(pred_img[0])

            images = [
                ("Sinogramma normalizzato", sino_plot),
                ("BackProjection", bp_plot),
                ("Predizione ViTRefiner", pred_plot),
            ]

            if target is not None:
                target_norm = minmax_scale(target, cfg.img_min, cfg.img_max)
                images.append(("Target", tensor_to_numpy(target_norm)))
            else:
                st.info("Target non trovato per il file selezionato.")

            st.caption(f"File: {sample_label}")
            plot_outputs(images, sinogram=True)

            if iter_imgs:
                total_steps = len(iter_imgs)
                iter_arrays: List[np.ndarray] = []
                iter_labels: List[str] = []
                for idx, step_tensor in enumerate(iter_imgs):
                    sample_tensor = step_tensor[0] if step_tensor.dim() > 0 else step_tensor
                    iter_arrays.append(tensor_to_numpy(sample_tensor))
                    if idx == 0:
                        label = f"Step {idx} (BP)"
                    elif idx == total_steps - 1:
                        label = f"Step {idx} (finale)"
                    else:
                        label = f"Step {idx}"
                    iter_labels.append(label)

                st.subheader("Iterazioni del modello")
                selected_label = st.selectbox(
                    "Seleziona uno step da visualizzare",
                    iter_labels,
                    index=len(iter_labels) - 1,
                )
                selected_idx = iter_labels.index(selected_label)
                st.image(
                    iter_arrays[selected_idx],
                    caption=selected_label,
                    use_column_width=True,
                    clamp=True,
                )
            else:
                st.info("Il modello non restituisce step intermedi da visualizzare.")
        except Exception as exc:
            st.error(f"Errore durante l'elaborazione: {exc}")
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

    else:
        st.info("Seleziona o carica un file per avviare l'inferenza.")


if __name__ == "__main__":
    main()
