"""Beamformer + transformer composite models."""
from typing import List

import torch
import torch.nn as nn

from .vit import ViTRefiner


class DelayAndSumTransformer(nn.Module):
    """Base model: single beamformer followed by ViT refinement."""

    def __init__(
        self,
        beamformer: nn.Module,
        vit: ViTRefiner,
        freeze_beamformer: bool = True,
    ) -> None:
        super().__init__()
        self.beamformer = beamformer
        self.vit = vit
        if freeze_beamformer:
            for p in self.beamformer.parameters():
                p.requires_grad = False

    def forward(self, sino: torch.Tensor):
        beamformed = self.beamformer(sino)
        out = self.vit(beamformed)
        initial_img = beamformed[:, :1, ...]
        intermediates = [out]
        return out, initial_img, intermediates


class UnrolledDelayAndSumTransformer(nn.Module):
    """Unrolled variant with iterative data-consistency steps."""

    def __init__(
        self,
        beamformer_module: nn.Module,
        forward_module: nn.Module,
        vit_module: ViTRefiner,
        num_steps: int,
        data_consistency_weight: float = 1.0,
        learnable_data_consistency_weight: bool = False,
        freeze_beamformer: bool = False,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1 for UnrolledDelayAndSumTransformer")
        self.beamformer = beamformer_module
        self.forward_operator = forward_module
        self.vit = vit_module
        self.num_steps = int(num_steps)
        weight = torch.tensor(float(data_consistency_weight), dtype=torch.float32)
        if learnable_data_consistency_weight:
            self.data_consistency_weight = nn.Parameter(weight)
        else:
            self.register_buffer("data_consistency_weight", weight)
        if freeze_beamformer:
            for p in self.beamformer.parameters():
                p.requires_grad = False

    def forward(self, sino: torch.Tensor):
        x0_full = self.beamformer(sino)
        sino_normalized = self.beamformer.normalize_with_cached_stats(sino)
        static_features = x0_full[:, 1:, :, :]
        if static_features.numel() == 0:
            static_features = None
        x0 = x0_full[:, :1, :, :]
        xi = x0
        intermediates: List[torch.Tensor] = []

        weight = self.data_consistency_weight.view(1, 1, 1, 1)

        for _ in range(self.num_steps):
            sino_est = self.forward_operator(xi)
            sino_residual = sino_normalized - sino_est
            correction = self.beamformer(
                sino_residual,
                update_cache=False,
                pre_normalized=True,
                return_magnitude=False,
            )
            xi = xi + weight * correction
            if static_features is not None:
                vit_input = torch.cat([xi, static_features], dim=1)
            else:
                vit_input = xi
            vit_out = self.vit(vit_input)
            xi = vit_out[:, :1, :, :]
            intermediates.append(xi)

        return xi, x0, intermediates
