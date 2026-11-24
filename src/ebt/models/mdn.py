"""Mixture density network (method.md section 16.8; Bishop 1994, NCRG/94/004, section 3).

The network maps x to the parameters of a mixture of m Gaussian kernels in the target y:
mixing coefficients by a softmax of the outputs (Bishop's eq. 25), widths by an exponential
(eq. 26) and centres directly (eq. 27). Its loss is the negative log-likelihood of the
targets (eq. 28 and 29).
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ebt.config import MLPConfig
from ebt.models.mlp import mlp


@dataclass
class Mixture:
    """Parameters of one Gaussian mixture per row, each of shape (B, m), for scalar targets."""

    log_weights: Tensor
    """log α_i, normalized by a log-softmax."""
    means: Tensor
    """μ_i."""
    log_scales: Tensor
    """log σ_i."""


class MixtureDensityNetwork(nn.Module):
    """An MLP with 3m outputs read as the parameters of an m-kernel mixture in y."""

    def __init__(self, context_dim: int, components: int, config: MLPConfig) -> None:
        super().__init__()
        self.components = components
        self.net = mlp(context_dim, 3 * components, config, output_bias=True)

    def forward(self, context: Tensor) -> Mixture:
        logits, means, log_scales = self.net(context).chunk(3, dim=-1)
        return Mixture(logits.log_softmax(dim=-1), means, log_scales)


def log_density(mixture: Mixture, target: Tensor) -> Tensor:
    """ln Σ_i α_i φ_i(y), with Gaussian kernels φ_i (eq. 22 and 23); target (B, 1) -> (B,)."""
    standardized = (target - mixture.means) / mixture.log_scales.exp()
    log_kernel = -0.5 * standardized**2 - mixture.log_scales - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(mixture.log_weights + log_kernel, dim=-1)


def negative_log_likelihood(mixture: Mixture, target: Tensor) -> Tensor:
    """Mean over rows of −ln Σ_i α_i φ_i(y) (eq. 28 and 29, divided by the number of rows)."""
    return -log_density(mixture, target).mean()


def most_probable_branch(mixture: Mixture) -> Tensor:
    """Centre of the kernel with the largest α_i (eq. 49), shape (B, 1).

    Bishop recommends this readout for multi-valued mappings with a finite number of distinct
    branches.
    """
    index = mixture.log_weights.argmax(dim=-1, keepdim=True)
    return mixture.means.gather(-1, index)


def most_probable_value(mixture: Mixture) -> Tensor:
    """Centre of the kernel with the largest α_i / σ_i^c (eq. 48), c = 1 for scalar targets."""
    index = (mixture.log_weights - mixture.log_scales).argmax(dim=-1, keepdim=True)
    return mixture.means.gather(-1, index)


def sampled_centre(mixture: Mixture, generator: torch.Generator) -> Tensor:
    """Centre of a kernel drawn with probability α_i, shape (B, 1)."""
    index = torch.multinomial(mixture.log_weights.exp(), 1, generator=generator)
    return mixture.means.gather(-1, index)
