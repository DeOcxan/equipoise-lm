"""Shared test support: small parameterized energies and float64 loop construction."""

import torch
from torch import Tensor, nn

from ebt.config import ThinkingConfig
from ebt.core.energy import EnergyModel
from ebt.core.thinking import ThinkingLoop
from ebt.tasks.mnist import ImageData, make_data


def float64_loop(config: ThinkingConfig) -> ThinkingLoop:
    """A ThinkingLoop whose α is created in float64, so that values such as 0.2 are exact
    to float64 precision rather than rounded to float32 first."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return ThinkingLoop(config)
    finally:
        torch.set_default_dtype(previous)


class TanhEnergy(EnergyModel):
    """E(x, y) = w_2 · tanh(W_1 [x; y] + b_1), applied independently to every leading index.

    Works for candidates (B, D) with context (B, C), and for sequences (B, S, D) with context
    (B, S, C), where it gives one energy per position. tanh gives non-zero third derivatives,
    so second-order gradients are exercised.
    """

    candidate_event_dims = 1

    def __init__(self, context_dim: int, candidate_dim: int, hidden: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.hidden = nn.Parameter(
            torch.randn(hidden, context_dim + candidate_dim, generator=generator) / 2
        )
        self.bias = nn.Parameter(torch.randn(hidden, generator=generator) / 2)
        self.output = nn.Parameter(torch.randn(hidden, generator=generator))

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        features = torch.cat([context, encoded], dim=-1)
        return torch.tanh(features @ self.hidden.T + self.bias) @ self.output


class TransformedTanhEnergy(TanhEnergy):
    """TanhEnergy with a learnable elementwise affine map after every step.

    Stands in for the text candidate RMSNorm (method.md section 4.1): a post-step transform
    with its own parameters.
    """

    def __init__(self, context_dim: int, candidate_dim: int, hidden: int, seed: int) -> None:
        super().__init__(context_dim, candidate_dim, hidden, seed)
        generator = torch.Generator().manual_seed(seed + 1)
        self.scale = nn.Parameter(1 + 0.1 * torch.randn(candidate_dim, generator=generator))
        self.shift = nn.Parameter(0.1 * torch.randn(candidate_dim, generator=generator))

    def after_step(self, candidate: Tensor) -> Tensor:
        return candidate * self.scale + self.shift


def synthetic_images() -> ImageData:
    """Bright squares at random positions on a dark background, 28 x 28 pixels."""
    generator = torch.Generator().manual_seed(0)
    images = torch.zeros(160, 28, 28, dtype=torch.uint8)
    corners = torch.randint(0, 20, (160, 2), generator=generator)
    for image, (row, column) in zip(images, corners.tolist(), strict=True):
        image[row : row + 8, column : column + 8] = 255
    return make_data(images[:128], images[128:], validation_size=32, seed=0)
