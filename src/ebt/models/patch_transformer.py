"""Patch-transformer energy and denoiser for images (method.md sections 3.2 and 17.3).

The design follows the authors' bidirectional image EBT (`model/bi_ebt_adaln.py` and
`model/img/ebt_denoise.py`), described in our own words:

- a convolutional patch embedding with kernel and stride equal to the patch size, plus fixed
  2-D sine-cosine positions;
- blocks of attention and an MLP (GELU, tanh approximation), each after a layer norm without
  affine parameters, modulated by shift, scale and residual-scale vectors computed from one
  condition vector (adaptive layer norm with zero initialization);
- one learned condition vector, the same for every image, since the denoising energy is
  unconditional;
- a final modulated layer norm and a linear head per patch.

The energy's head maps each patch to one scalar without bias and the energy is the mean over
patches. The denoiser's head maps each patch to its pixels, with bias.
"""

import math

import torch
from torch import Tensor, nn

from ebt.config import PatchTransformerConfig
from ebt.core.energy import EnergyModel
from ebt.models.attention import attention

LAYER_NORM_EPS = 1e-6


def sincos_positions(dim: int, grid: int) -> Tensor:
    """Fixed 2-D positions, shape (grid², dim), patches in row-major order.

    The first dim / 2 channels encode the column and the last dim / 2 the row. Each half is
    [sin(p ω_j), cos(p ω_j)] with ω_j = 10000^(−j / (dim / 4)), j < dim / 4.
    """

    def axis(position: Tensor, channels: int) -> Tensor:
        omega = 1.0 / 10000 ** (torch.arange(channels // 2, dtype=torch.float64) / (channels // 2))
        angles = position.unsqueeze(-1) * omega
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    index = torch.arange(grid, dtype=torch.float64)
    row, column = torch.meshgrid(index, index, indexing="ij")
    half = dim // 2
    return torch.cat([axis(column.flatten(), half), axis(row.flatten(), half)], dim=-1).float()


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """x · (1 + scale) + shift, with one shift and scale per image broadcast over patches."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the patches of each image, using `attention`."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        out = attention(query, key, value)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    """x ← x + r₁ · attn(modulate(norm(x))), then x ← x + r₂ · mlp(modulate(norm(x))).

    The shifts, scales and residual scales r are computed from the condition vector.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        modulation = self.modulation(condition).chunk(6, dim=-1)
        shift1, scale1, residual1, shift2, scale2, residual2 = modulation
        x = x + residual1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        return x + residual2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class PatchBackbone(nn.Module):
    """Images (B, C, H, W) -> per-patch outputs (B, patches, outputs)."""

    def __init__(self, config: PatchTransformerConfig, outputs: int, output_bias: bool) -> None:
        super().__init__()
        self.config = config
        self.grid = config.image_size // config.patch
        dim = config.dim
        self.embed = nn.Conv2d(config.channels, dim, config.patch, stride=config.patch)
        self.register_buffer("positions", sincos_positions(dim, self.grid).unsqueeze(0))
        self.condition = nn.Parameter(torch.randn(dim))
        self.blocks = nn.ModuleList(
            Block(dim, config.heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, outputs, bias=output_bias)
        self._initialize()

    def _initialize(self) -> None:
        """Xavier-uniform linear and patch-embedding weights with zero biases; zero modulation
        and zero head, so that every block starts as the identity and the output as zero.
        The condition vector is drawn from N(0, 1), the default of a learned embedding."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.embed.weight.view(self.embed.weight.shape[0], -1))
        nn.init.zeros_(self.embed.bias)
        for modulation in [block.modulation for block in self.blocks] + [self.final_modulation]:
            nn.init.zeros_(modulation[-1].weight)
            nn.init.zeros_(modulation[-1].bias)
        nn.init.zeros_(self.head.weight)

    def forward(self, images: Tensor) -> Tensor:
        x = self.embed(images).flatten(2).transpose(1, 2) + self.positions
        condition = self.condition.expand(images.shape[0], -1)
        for block in self.blocks:
            x = block(x, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.head(modulate(self.final_norm(x), shift, scale))


class PatchEnergy(EnergyModel):
    """Unconditional image energy: the mean over patches of one scalar per patch."""

    candidate_event_dims = 3

    def __init__(self, config: PatchTransformerConfig) -> None:
        super().__init__()
        self.net = PatchBackbone(config, outputs=1, output_bias=False)

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        """Energy of each image, shape (B,); `context` is unused (method.md section 3.2)."""
        return self.net(encoded).squeeze(-1).mean(dim=1)


class PatchDenoiser(nn.Module):
    """Feed-forward baseline: predicts the clean image from the noisy one in one pass."""

    def __init__(self, config: PatchTransformerConfig) -> None:
        super().__init__()
        self.net = PatchBackbone(
            config, outputs=config.patch**2 * config.channels, output_bias=True
        )

    def forward(self, images: Tensor) -> Tensor:
        return unpatchify(self.net(images), self.net.config)


def unpatchify(patches: Tensor, config: PatchTransformerConfig) -> Tensor:
    """(B, patches, patch · patch · C) -> (B, C, H, W).

    Patches are in row-major order and each patch's values in (row, column, channel) order.
    """
    batch, p, c = patches.shape[0], config.patch, config.channels
    grid = math.isqrt(patches.shape[1])
    x = patches.reshape(batch, grid, grid, p, p, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(batch, c, grid * p, grid * p)
