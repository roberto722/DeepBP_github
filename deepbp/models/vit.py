"""Lightweight ViT-like refiner and helpers."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_ch: int = 1, embed_dim: int = 256, patch: int = 16, stride: Optional[int] = None) -> None:
        super().__init__()
        self.patch = patch
        self.patch_size = _pair(patch)
        self.stride = patch if stride is None else stride
        self.stride_size = _pair(self.stride)
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride_size,
        )

    def compute_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        ph, pw = self.patch_size
        sh, sw = self.stride_size
        Hp = max((H - ph) // sh + 1, 0)
        Wp = max((W - pw) // sw + 1, 0)
        return Hp, Wp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        B, _, H, W = x.shape
        x = self.proj(x)
        _, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp), (H, W), self.stride_size


class TransformerEncoder(nn.Module):
    """A small stack of standard Transformer encoder blocks."""

    def __init__(self, dim: int = 256, depth: int = 6, heads: int = 8, mlp_ratio: float = 4.0, p_drop: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=p_drop, batch_first=True),
                        "ln2": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, int(dim * mlp_ratio)),
                            nn.GELU(),
                            nn.Linear(int(dim * mlp_ratio), dim),
                        ),
                    }
                )
            )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            y = blk["ln1"](x)
            y, _ = blk["attn"](y, y, y, need_weights=False)
            x = x + self.drop(y)
            y = blk["ln2"](x)
            y = blk["mlp"](y)
            x = x + self.drop(y)
        return x


class PatchDecoder(nn.Module):
    """Decode token features on the patch grid with local convolutions."""

    def __init__(self, embed_dim: int, patch_area: int, num_blocks: int = 3) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("PatchDecoder requires at least one convolutional block")

        layers: List[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv2d(embed_dim, patch_area, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layers(x)
        x = x + residual
        x = self.proj(x)
        return x


class LocalFusionBlock(nn.Module):
    """Fuse overlapping patches on the full-resolution image with local smoothing."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class ViTRefiner(nn.Module):
    """Simple ViT-like refiner that denoises/refines the beamformed image."""

    def __init__(
        self,
        in_ch: int = 1,
        embed_dim: int = 256,
        patch: int = 16,
        stride: Optional[int] = None,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.stride = patch if stride is None else stride
        self.patch_size = _pair(self.patch)
        self.stride_size = _pair(self.stride)
        self.embed = PatchEmbed(in_ch, embed_dim, patch, stride=self.stride)
        self.embed_dim = embed_dim
        self.pos_embed: Optional[nn.Parameter] = None
        self.grid_size: Optional[Tuple[int, int]] = None
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio, p_drop)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.token_decoder = PatchDecoder(embed_dim, self.patch_area, num_blocks=3)
        fusion_hidden = max(embed_dim // 8, 16)
        self.local_fusion = LocalFusionBlock(in_channels=1, hidden_channels=fusion_hidden)
        self.register_buffer("fold_weight", torch.empty(0), persistent=False)
        self._fold_weight_shape: Optional[Tuple[int, int]] = None
        self._fold_weight_stride: Optional[Tuple[int, int]] = None

    def _compute_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        return self.embed.compute_grid_size(H, W)

    def _build_pos_embed(
        self,
        H: int,
        W: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        Hp, Wp = self._compute_grid_size(H, W)
        pe = torch.zeros(1, Hp * Wp, dim, device=device, dtype=dtype)
        pe = nn.Parameter(pe)
        nn.init.trunc_normal_(pe, std=0.02)
        self.pos_embed = pe
        self.grid_size = (Hp, Wp)

    def _get_fold_weight(
        self,
        shape: Tuple[int, int],
        stride: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = self.fold_weight
        if (
            weight.numel() == 0
            or self._fold_weight_shape != shape
            or self._fold_weight_stride != stride
        ):
            ones = torch.ones(1, 1, shape[0], shape[1], device=device, dtype=dtype)
            weight = F.fold(
                F.unfold(ones, kernel_size=self.patch_size, stride=stride),
                output_size=shape,
                kernel_size=self.patch_size,
                stride=stride,
            )
            weight = weight.to(device=device, dtype=dtype)
            self.fold_weight = weight
            self._fold_weight_shape = shape
            self._fold_weight_stride = stride
        elif weight.device != device or weight.dtype != dtype:
            weight = weight.to(device=device, dtype=dtype)
            self.fold_weight = weight
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, (Hp, Wp), (H, W), stride = self.embed(x)
        stride_hw = tuple(stride)
        expected_grid = self._compute_grid_size(H, W)
        if expected_grid != (Hp, Wp):
            Hp, Wp = expected_grid
        B, N, D = z.shape
        if Hp * Wp != N:
            raise ValueError(f"Token count ({N}) does not match grid size ({Hp}x{Wp}).")
        if (
            self.pos_embed is None
            or self.pos_embed.shape[1] != N
            or self.pos_embed.shape[2] != D
            or self.grid_size != (Hp, Wp)
        ):
            self._build_pos_embed(H, W, D, x.device, dtype=z.dtype)
        if self.pos_embed is None:
            raise RuntimeError("Failed to initialize positional embeddings")
        pe = self.pos_embed
        if pe.device != z.device or pe.dtype != z.dtype:
            pe = pe.to(device=z.device, dtype=z.dtype)
            self.pos_embed = nn.Parameter(pe)
        z = z + pe
        z = self.encoder(z)
        z = z.transpose(1, 2).reshape(B, D, Hp, Wp)
        pixels = self.token_decoder(z)
        pixels = pixels.reshape(B, self.patch_area, Hp * Wp)
        img = F.fold(
            pixels,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=stride_hw,
        )
        norm = self._get_fold_weight((H, W), stride_hw, x.device, x.dtype)
        img = img / norm.clamp_min(1.0)
        img = self.local_fusion(img)
        if img.shape == x.shape:
            img = img + x
        return img


def adapt_vitrefiner_state_dict(state_dict: dict) -> Tuple[dict, bool]:
    """Upgrade legacy ViTRefiner checkpoints to the new convolutional decoder."""

    new_state = dict(state_dict)
    converted = False

    suffix_weight = "proj_out.weight"
    suffix_bias = "proj_out.bias"

    for key in list(state_dict.keys()):
        if key.endswith(suffix_weight):
            prefix = key[: -len(suffix_weight)]
            bias_key = prefix + suffix_bias
            weight = new_state.pop(key)
            bias = new_state.pop(bias_key, None)

            conv_weight = weight.new_zeros((weight.shape[0], weight.shape[1], 3, 3))
            conv_weight[:, :, 1, 1] = weight

            new_state[prefix + "token_decoder.proj.weight"] = conv_weight
            if bias is not None:
                new_state[prefix + "token_decoder.proj.bias"] = bias
            else:
                new_state[prefix + "token_decoder.proj.bias"] = weight.new_zeros(weight.shape[0])
            converted = True
        elif key.endswith("out_conv.weight") or key.endswith("out_conv.bias"):
            new_state.pop(key)
            converted = True

    return new_state, converted
