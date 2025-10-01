"""Streamlit dashboard per esplorare il modello BeamformerTransformer."""
import json
import os
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import load_hdf5_sample, minmax_scale
from main import (
    TrainConfig,
    create_model,
    load_checkpoint,
    psnr,
    ssim,
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
    st.set_page_config(page_title="Deep Beamformer Dashboard", layout="wide")
    st.title("Dashboard Deep Beamformer")

    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"Dispositivo: {device}")

    default_work_dir = Path(cfg.work_dir).resolve()
    parent_work_dir = default_work_dir.parent
    run_directories: List[Path] = []
    if parent_work_dir.is_dir():
        run_directories = sorted(d for d in parent_work_dir.iterdir() if d.is_dir())
    else:
        st.sidebar.warning(
            f"Cartella degli esperimenti non trovata: {parent_work_dir}."
        )

    selected_run_path = default_work_dir
    if run_directories:
        run_names = [d.name for d in run_directories]
        default_index = run_names.index(default_work_dir.name) if default_work_dir.name in run_names else 0
        selected_run_name = st.sidebar.selectbox(
            "Esperimento",
            run_names,
            index=default_index if run_names else 0,
        )
        selected_run_path = parent_work_dir / selected_run_name
    else:
        st.sidebar.info(
            "Nessun esperimento trovato nella cartella padre."
        )

    config_dict: Optional[dict] = None
    config_source: Optional[Tuple[str, str]] = None
    config_issue: Optional[str] = None
    config_path = selected_run_path / "config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            if isinstance(loaded_config, dict):
                config_dict = loaded_config
                config_source = ("file", config_path.name)
            else:
                config_issue = (
                    f"Il file {config_path.name} non contiene una configurazione valida."
                )
        except (OSError, json.JSONDecodeError) as exc:
            config_issue = f"Errore nella lettura di {config_path.name}: {exc}"

    ckpt_fallback_errors: List[str] = []
    if config_dict is None and selected_run_path.is_dir():
        ckpt_files = [p for p in selected_run_path.glob("*.pt") if p.is_file()]
        ckpt_name_to_path = {p.name: p for p in ckpt_files}
        ckpt_names_sorted = sorted(ckpt_name_to_path.keys())
        ckpts_priority = [
            name for name in ("best.pt", "last.pt") if name in ckpt_name_to_path
        ]
        ckpt_candidates = ckpts_priority + [
            name for name in ckpt_names_sorted if name not in ckpts_priority
        ]
        for ckpt_name in ckpt_candidates:
            ckpt_path = ckpt_name_to_path[ckpt_name]
            try:
                payload = torch.load(ckpt_path, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                ckpt_fallback_errors.append(f"{ckpt_name}: {exc}")
                continue
            if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
                config_dict = payload["config"]
                config_source = ("checkpoint", ckpt_name)
                break

    if config_dict is not None:
        cfg_field_names = {field.name for field in fields(TrainConfig)}
        filtered_config = {
            key: value for key, value in config_dict.items() if key in cfg_field_names
        }
        missing_keys = sorted(cfg_field_names - filtered_config.keys())
        try:
            cfg = TrainConfig(**filtered_config)
        except TypeError as exc:
            st.sidebar.warning(
                "Configurazione salvata non valida; verranno usati i valori di default. "
                f"Dettagli: {exc}"
            )
            cfg = TrainConfig()
        else:
            if config_source is not None:
                if config_source[0] == "file":
                    st.sidebar.info(
                        f"Configurazione caricata da {config_source[1]}."
                    )
                else:
                    st.sidebar.info(
                        f"Configurazione recuperata dal checkpoint {config_source[1]}."
                    )
            if config_issue is not None:
                st.sidebar.warning(config_issue)
            if missing_keys:
                st.sidebar.warning(
                    "Parametri mancanti nella configurazione salvata "
                    f"({', '.join(missing_keys)}); sono stati usati i valori di default."
                )
    else:
        if config_issue is not None:
            st.sidebar.warning(
                f"{config_issue} Verranno usati i valori di default."
            )
        elif not config_path.exists():
            st.sidebar.warning(
                "Configurazione non trovata per l'esperimento selezionato; verranno usati i valori di default."
            )
        elif ckpt_fallback_errors:
            st.sidebar.warning(
                "Impossibile recuperare la configurazione salvata dai checkpoint; verranno usati i valori di default."
            )
        else:
            st.sidebar.warning(
                "Impossibile recuperare la configurazione salvata; verranno usati i valori di default."
            )
        cfg = TrainConfig()
        if ckpt_fallback_errors:
            st.sidebar.caption(
                "Dettagli: " + "; ".join(ckpt_fallback_errors[:1])
            )

    cfg.work_dir = str(selected_run_path.resolve())
    work_dir = cfg.work_dir
    st.sidebar.caption(f"Cartella esperimento: {work_dir}")

    ckpt_dir = Path(work_dir)
    ckpt_names: List[str] = []
    if ckpt_dir.is_dir():
        ckpt_paths = sorted(p for p in ckpt_dir.glob("*.pt") if p.is_file())
        ckpt_names = [p.name for p in ckpt_paths]
    ckpts_priority = [name for name in ("best.pt", "last.pt") if name in ckpt_names]
    remaining_ckpts = sorted(name for name in ckpt_names if name not in ckpts_priority)
    ckpts = ckpts_priority + remaining_ckpts

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
            "Nessun checkpoint trovato nella cartella selezionata. "
            "Verranno usati pesi random inizializzati."
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

            (
                sino_norm,
                initial_img,
                pred_img,
                iter_imgs,
                beamformer_stack,
            ) = run_inference_steps(
                model,
                sinogram_raw,
                cfg,
                device=device,
                normalize=True,
            )

            sino_plot = tensor_to_numpy(sino_norm[0])
            initial_plot = (
                tensor_to_numpy(initial_img[0]) if initial_img is not None else None
            )
            pred_plot = tensor_to_numpy(pred_img[0])

            images: List[Tuple[str, np.ndarray]] = [
                ("Sinogramma normalizzato", sino_plot),
            ]
            if initial_plot is not None:
                images.append(("Output beamformer", initial_plot))
            images.append(("Predizione ViTRefiner", pred_plot))

            metrics_values = None
            if target is not None:
                target_norm = minmax_scale(target, cfg.img_min, cfg.img_max).to(
                    pred_img.dtype
                )

                batch_size = pred_img.shape[0]

                def to_bchw_clamped(t: torch.Tensor) -> torch.Tensor:
                    tensor = t.to(dtype=pred_img.dtype)
                    while tensor.dim() < 4:
                        if tensor.dim() == 2:
                            tensor = tensor.unsqueeze(0).unsqueeze(0)
                        elif tensor.dim() == 3:
                            if tensor.shape[0] == batch_size:
                                tensor = tensor.unsqueeze(1)
                            else:
                                tensor = tensor.unsqueeze(0)
                        else:
                            tensor = tensor.unsqueeze(0)
                    return tensor.clamp(0.0, 1.0)

                pred_bchw = to_bchw_clamped(pred_img)
                target_bchw = to_bchw_clamped(target_norm)

                psnr_value = float(psnr(pred_bchw, target_bchw).reshape(-1)[0].cpu())
                ssim_value = float(ssim(pred_bchw, target_bchw).reshape(-1)[0].cpu())
                mae_value = float(F.l1_loss(pred_bchw, target_bchw).cpu())

                metrics_values = {
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "mae": mae_value,
                }

                images.append(("Target", tensor_to_numpy(target_norm)))
            else:
                st.info("Target non trovato per il file selezionato.")

            st.caption(f"File: {sample_label}")
            if metrics_values is not None:
                psnr_col, ssim_col, mae_col = st.columns(3)
                psnr_col.metric("PSNR", f"{metrics_values['psnr']:.2f} dB")
                ssim_col.metric("SSIM", f"{metrics_values['ssim']:.4f}")
                mae_col.metric("MAE", f"{metrics_values['mae']:.4f}")
            plot_outputs(images, sinogram=True)

            has_initial = initial_img is not None
            if iter_imgs:
                total_steps = len(iter_imgs)
                iter_arrays: List[np.ndarray] = []
                iter_labels: List[str] = []
                for idx, step_tensor in enumerate(iter_imgs):
                    sample_tensor = step_tensor[0] if step_tensor.dim() > 0 else step_tensor
                    iter_arrays.append(tensor_to_numpy(sample_tensor))
                    if idx == 0 and has_initial:
                        label = f"Step {idx} (beamformer)"
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

            if beamformer_stack is not None and beamformer_stack.shape[1] > 1:
                st.info(
                    "Il beamformer ha fornito più canali. È possibile ispezionare tutte le feature "
                    "dalla variabile `beamformer_stack`."
                )
        except Exception as exc:
            st.error(f"Errore durante l'elaborazione: {exc}")
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

    else:
        st.info("Seleziona o carica un file per avviare l'inferenza.")


if __name__ == "__main__":
    main()
