"""Derivative-free optimizer of Florence et al. (2021), arXiv:2109.00137, Algorithm 1.

Used for inference with the implicit behavioural cloning energy (method.md section 16.8).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel


@dataclass(frozen=True)
class DFOConfig:
    """Settings of Algorithm 1; the defaults are the values of Appendix B.1 of the paper."""

    samples: int = 16_384
    iterations: int = 3
    initial_std: float = 0.33
    shrink: float = 0.5
    """K, the factor applied to the noise scale after every resampling."""


def derivative_free_optimization(
    model: EnergyModel,
    context: Tensor,
    bounds: tuple[float, float],
    config: DFOConfig,
    generator: torch.Generator,
) -> tuple[Tensor, PassCount]:
    """Algorithm 1 for scalar candidates; context (B, C) -> candidates (B, 1) and the passes.

    1. Draw `samples` candidates per row uniformly from `bounds`.
    2. At every iteration, compute their energies and p̃ = softmax(−E). Except at the last
       iteration, resample the candidates with replacement in proportion to p̃, add Gaussian
       noise of scale σ, clip to `bounds` and multiply σ by K.
    3. Return the candidate with the largest p̃ at the last iteration.

    Gradients are not used; every iteration costs one forward pass per sample.
    """
    low, high = bounds
    rows, count = context.shape[0], config.samples
    options = {"dtype": context.dtype, "device": context.device}
    with torch.no_grad():
        candidates = low + (high - low) * torch.rand(rows, count, 1, generator=generator, **options)
        expanded = context.unsqueeze(1).expand(-1, count, *context.shape[1:])
        std = config.initial_std
        for iteration in range(config.iterations):
            energy = model.energy(expanded, model.encode_candidate(candidates))
            probabilities = torch.softmax(-energy, dim=-1)
            if iteration < config.iterations - 1:
                index = torch.multinomial(
                    probabilities, count, replacement=True, generator=generator
                )
                candidates = candidates.gather(1, index.unsqueeze(-1))
                noise = torch.randn(candidates.shape, generator=generator, **options)
                candidates = (candidates + std * noise).clamp(low, high)
                std *= config.shrink
        best = probabilities.argmax(dim=-1)
        chosen = candidates[torch.arange(rows, device=context.device), best]
    return chosen, PassCount(forward=config.iterations * count) * rows
