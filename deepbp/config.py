"""Training configuration and model factory helpers."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

import torch

from .geometry import LinearProbeGeom, build_delay_and_sum_lut
from .beamformers.delay_and_sum import DelayAndSumLinear, ForwardProjectionLinear
from .beamformers.fk import FkMigrationLinear, ForwardProjectionFk
from .models.vit import ViTRefiner
from .models.transformer import DelayAndSumTransformer, UnrolledDelayAndSumTransformer


@dataclass
class TrainConfig:
    """High-level configuration driving geometry, model and training hyper-parameters."""

    # Geometry / sampling
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
    trainable_apodization: bool = True
    beamformer_type: Literal["das", "fk"] = "fk"
    fk_fft_pad: int = 20
    fk_window: Optional[str] = None
    fk_learnable_output_normalization: bool = False
    fk_output_normalization_scale: Optional[float] = 1
    fk_output_normalization_shift: Optional[float] = 0
    fk_output_norm_scale_init: Optional[float] = None
    fk_output_norm_shift_init: Optional[float] = None
    fk_output_components: Tuple[str, ...] = ("magnitude", "real", "imag")

    # ViT refiner
    vit_patch: int = 16
    vit_stride: Optional[int] = None

    # Training
    epochs: int = 300
    batch_size: int = 8
    lr: float = 5e-4
    num_workers: int = 4
    clip_grad: float = 100.0
    use_tqdm: bool = True
    weight_alpha: float = 0
    weight_threshold: Optional[float] = None
    ssim_mask_threshold: Optional[float] = 0.5
    ssim_mask_dilation: int = 0
    normalize_targets: bool = False

    # Model variants
    model_variant: str = "unrolled"
    unroll_steps: int = 5
    data_consistency_weight: float = 1.0
    learnable_data_consistency_weight: bool = True

    # Global scaling (per-domain)
    # sino_min: float = -11.0322
    # sino_max: float = 12.5394
    # img_min: float = 0.0
    # img_max: float = 316.9658

    sino_min: float = -1
    sino_max: float = 1
    img_min: float = 0.0
    img_max: float = 1.5

    # Paths / dataset
    work_dir: str = "./runs/VOC_DAS_transformer"
    data_root: str = "E:/Scardigno/datasets_transformer_proj"
    sino_dir: str = "E:\Scardigno\Fotoacustica\dataset\VOC_forearm_2000" # "Forearm2000_hdf5/train_val_tst"
    recs_dir: str = "Forearm2000_recs/L1_Shearlet"  # NOT USED IN VOC
    dataset_type: Literal["hdf5", "voc"] = "voc"
    """Dataset backend to use for loading sinograms/reconstructions."""
    save_val_images: bool = True
    max_val_images: int = 1
    val_intermediate_indices: Optional[List[int]] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]
    )
    resume_training: bool = False
    resume_checkpoint: Optional[str] = None
    pretrained_checkpoint: Optional[str] = None
    pretrained_load_optimizer_state: bool = False


def build_geometry(cfg: TrainConfig) -> LinearProbeGeom:
    """Instantiate the acquisition geometry from the provided configuration."""

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


def build_projection_operators(
    cfg: TrainConfig,
    device: torch.device,
    trainable_apodization: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Construct beamformer and forward operators according to configuration."""

    geom = build_geometry(cfg)
    beamformer_type = cfg.beamformer_type.lower()

    if beamformer_type == "das":
        lut = build_delay_and_sum_lut(geom, device=device)
        beamformer = DelayAndSumLinear(
            geom,
            lut,
            trainable_apodization=trainable_apodization,
        )
        forward_op = ForwardProjectionLinear(beamformer)
    elif beamformer_type == "fk":
        fft_pad = cfg.fk_fft_pad if cfg.fk_fft_pad is not None else 0
        beamformer = FkMigrationLinear(
            geom,
            trainable_apodization=trainable_apodization,
            per_channel_apodization=False,
            fft_pad=fft_pad,
            window=cfg.fk_window,
            learnable_output_normalization=cfg.fk_learnable_output_normalization,
            static_output_scale=cfg.fk_output_normalization_scale,
            static_output_shift=cfg.fk_output_normalization_shift,
            output_norm_scale_init=cfg.fk_output_norm_scale_init,
            output_norm_shift_init=cfg.fk_output_norm_shift_init,
            output_components=cfg.fk_output_components,
        )
        forward_op = ForwardProjectionFk(beamformer)
    else:
        raise ValueError(
            f"Unsupported beamformer_type '{cfg.beamformer_type}'. Expected 'das' or 'fk'."
        )

    return beamformer, forward_op


def create_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Build beamformer + ViT refiner according to the provided configuration."""

    beamformer, forward_op = build_projection_operators(
        cfg,
        device,
        trainable_apodization=cfg.trainable_apodization,
    )

    vit_stride = cfg.vit_stride if cfg.vit_stride is not None else cfg.vit_patch
    vit_in_ch = getattr(beamformer, "default_output_channels", 1)
    vit = ViTRefiner(
        in_ch=vit_in_ch,
        embed_dim=256,
        patch=cfg.vit_patch,
        stride=vit_stride,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        p_drop=0.1,
    )
    Hp, Wp = vit.embed.compute_grid_size(cfg.ny, cfg.nx)
    ph, pw = vit.patch_size
    sh, sw = vit.stride_size
    if Hp <= 0 or Wp <= 0:
        raise ValueError(
            "Image dimensions must be compatible with ViT patch and stride. "
            f"Got (ny={cfg.ny}, nx={cfg.nx}) with patch={vit.patch_size} and stride={vit.stride_size}."
        )
    if (cfg.ny - ph) % sh != 0 or (cfg.nx - pw) % sw != 0:
        raise ValueError(
            "Image dimensions must align with patch/stride grid. "
            f"Got (ny={cfg.ny}, nx={cfg.nx}), patch={vit.patch_size}, stride={vit.stride_size}."
        )
    vit._build_pos_embed(cfg.ny, cfg.nx, vit.embed_dim, device, dtype=torch.float32)
    freeze_beamformer = not cfg.trainable_apodization
    variant = cfg.model_variant.lower()
    if variant == "unrolled":
        if cfg.unroll_steps < 1:
            raise ValueError("TrainConfig.unroll_steps must be >= 1 for the unrolled variant")
        model = UnrolledDelayAndSumTransformer(
            beamformer,
            forward_op,
            vit,
            num_steps=cfg.unroll_steps,
            data_consistency_weight=cfg.data_consistency_weight,
            learnable_data_consistency_weight=cfg.learnable_data_consistency_weight,
            freeze_beamformer=freeze_beamformer,
        )
    elif variant == "base":
        model = DelayAndSumTransformer(
            beamformer,
            vit,
            freeze_beamformer=freeze_beamformer,
        )
    else:
        raise ValueError(
            f"Unsupported model_variant '{cfg.model_variant}'. Expected 'unrolled' or 'base'."
        )

    return model.to(device)
