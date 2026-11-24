"""Landscape-conditioned energy and noise schedule of IRED (method.md section 16.8).

Du et al. (2024), "Learning Iterative Reasoning through Energy Diffusion", arXiv:2406.11179,
code `yilundu/ired_code_release` (MIT). The architecture follows the authors' `models.py: EBM`,
scaled down to the parameter budget of the G2 task; the schedule follows their
`diffusion_lib/denoising_diffusion_pytorch_1d.py: cosine_beta_schedule`. Both are described in
our own words.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebt.core.energy import EnergyModel

COSINE_OFFSET = 0.008
"""s in the cosine schedule, the authors' default."""
MAX_BETA = 0.999


@dataclass(frozen=True)
class LandscapeSchedule:
    """Noise level of each of the K energy landscapes, k = 0 (least noise) to K − 1."""

    betas: Tensor
    alpha_bar: Tensor
    """ᾱ_k = Π_{j ≤ k} (1 − β_j); landscape k models √ᾱ_k · y + √(1 − ᾱ_k) · ε."""
    step_size: Tensor
    """λ_k = β_k / √(1 − ᾱ_k), the gradient step size on landscape k (the authors'
    `opt_step_size`)."""

    @property
    def landscapes(self) -> int:
        return len(self.betas)


def cosine_schedule(landscapes: int) -> LandscapeSchedule:
    """Cosine schedule: ᾱ(t) ∝ cos²(((t / K) + s) / (1 + s) · π / 2), t = 0..K.

    β_k = 1 − ᾱ(k + 1) / ᾱ(k), clipped to [0, 0.999]; ᾱ_k is the cumulative product of
    1 − β. Computed in float64 and returned in float32, as in the authors' code.
    """
    t = torch.linspace(0, landscapes, landscapes + 1, dtype=torch.float64)
    curve = torch.cos((t / landscapes + COSINE_OFFSET) / (1 + COSINE_OFFSET) * math.pi / 2) ** 2
    curve = curve / curve[0]
    betas = (1 - curve[1:] / curve[:-1]).clamp(0, MAX_BETA)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    step_size = betas / torch.sqrt(1 - alpha_bar)
    return LandscapeSchedule(betas.float(), alpha_bar.float(), step_size.float())


def sinusoidal_embedding(index: Tensor, dim: int) -> Tensor:
    """[sin(k ω_j), cos(k ω_j)] with ω_j = 10000^(−j / (dim/2 − 1)), j < dim / 2."""
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=index.device, dtype=index.dtype) * -(math.log(10000) / (half - 1))
    )
    angles = index.unsqueeze(-1) * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class LandscapeEnergy(EnergyModel):
    """E(x, y; k) = ‖fc4(h)‖² with h from three swish layers, the last two modulated by k.

    The context is [x, k]: the input followed by the landscape index as a float, so that the
    energy contract E(context, candidate) is unchanged. The index enters through a sinusoidal
    embedding and an MLP, which give a gain and a bias for layers 2 and 3:
    h ← swish(W h · (1 + gain) + bias).
    """

    candidate_event_dims = 1

    def __init__(
        self, context_dim: int, candidate_dim: int, hidden: int, embedding: int, time: int
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.time = nn.Sequential(nn.Linear(embedding, time), nn.GELU(), nn.Linear(time, time))
        self.fc1 = nn.Linear(context_dim + candidate_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, candidate_dim)
        self.film2 = nn.Linear(time, 2 * hidden)
        self.film3 = nn.Linear(time, 2 * hidden)

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        """Energy of each row, shape (*leading,), for context (*leading, C + 1)."""
        x, index = context[..., :-1], context[..., -1]
        time = self.time(sinusoidal_embedding(index, self.embedding))
        gain2, bias2 = self.film2(time).chunk(2, dim=-1)
        gain3, bias3 = self.film3(time).chunk(2, dim=-1)
        h = F.silu(self.fc1(torch.cat([x, encoded], dim=-1)))
        h = F.silu(self.fc2(h) * (gain2 + 1) + bias2)
        h = F.silu(self.fc3(h) * (gain3 + 1) + bias3)
        return self.fc4(h).pow(2).sum(dim=-1)


def landscape_context(x: Tensor, index: Tensor | int) -> Tensor:
    """[x, k] for every row: the context of `LandscapeEnergy`."""
    column = torch.as_tensor(index, dtype=x.dtype, device=x.device).expand(x.shape[0])
    return torch.cat([x, column.unsqueeze(-1)], dim=-1)


class OnLandscape(EnergyModel):
    """E(x, y; k) of `LandscapeEnergy` on one fixed landscape k, with context x alone."""

    candidate_event_dims = 1

    def __init__(self, inner: LandscapeEnergy, landscape: int) -> None:
        super().__init__()
        self.inner = inner
        self.landscape = landscape

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        return self.inner.energy(landscape_context(context, self.landscape), encoded)
