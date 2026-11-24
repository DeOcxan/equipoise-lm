"""IRED training loss (method.md section 16.8; Du et al. 2024, arXiv:2406.11179).

Follows the authors' `GaussianDiffusion1D.p_losses` for continuous tasks, described in our
own words: a denoising loss on the gradient of the energy, plus a contrastive loss between a
noised target and a noised refined negative on the same landscape and with the same noise.
"""

import torch
from torch import Tensor

from ebt.core.contrastive import contrastive_loss
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.inference.annealed import CLIP, LandscapeSteps, descend
from ebt.models.ired import LandscapeEnergy, landscape_context
from ebt.training.batch import Batch
from ebt.training.objectives import BatchLoss, TrainingObjective

CONTRASTIVE_WEIGHT = 0.5
"""The authors' `loss_scale` for continuous tasks."""
NEGATIVE_NOISE_SCALE = 3.0
"""Negatives start from the target noised with three times the noise."""
NEGATIVE_STEPS = 2
"""Accept-only-lower steps that refine a negative."""


def denoising_loss(model: EnergyModel, context: Tensor, noisy: Tensor, noise: Tensor) -> Tensor:
    """‖∇_y E(x, ỹ; k) − ε‖² averaged over rows and target dimensions (Eq. 5 of the paper).

    The gradient keeps its graph, so the loss's parameter gradient is second order.
    """
    point = noisy.detach().requires_grad_(True)
    energy = model.energy(context, model.encode_candidate(point))
    (gradient,) = torch.autograd.grad(energy.sum(), point, create_graph=True)
    return ((gradient - noise) ** 2).mean()


class IREDObjective(TrainingObjective):
    """Loss of one batch of standardized (x, y) rows.

    For every row a landscape k is drawn uniformly from 0..K − 1. With ε, ε' ~ N(0, I):

    1. Denoising: ‖∇_y E(x, √ᾱ_k y + √(1 − ᾱ_k) ε; k) − ε‖², averaged over rows and target
       dimensions (Eq. 5 of the paper). Its gradient with respect to θ is second order.
    2. Negative: start from √ᾱ_k y + √(1 − ᾱ_k) · 3ε', take NEGATIVE_STEPS accept-only-lower
       steps on landscape k without a graph, divide by √ᾱ_k, clamp to ±CLIP and re-noise with
       the same ε': y⁻ = √ᾱ_k ŷ⁻ + √(1 − ᾱ_k) ε'.
    3. Contrastive: softplus(E(x, y⁺; k) − E(x, y⁻; k)) with y⁺ = √ᾱ_k y + √(1 − ᾱ_k) ε', the
       two-class cross-entropy with the positive as the correct class.

    The loss is denoising + CONTRASTIVE_WEIGHT · contrastive.
    """

    def __init__(self, model: LandscapeEnergy, steps: LandscapeSteps) -> None:
        self.model = model
        self.steps = steps
        self.loop = None
        self.module = model

    def batch_loss(self, batch: Batch, generator: torch.Generator) -> BatchLoss:
        x, y = batch.context, batch.target
        rows, options = len(batch), {"dtype": y.dtype, "device": y.device}
        schedule = self.steps.schedule
        landscape = torch.randint(
            0, schedule.landscapes, (rows,), generator=generator, device=y.device
        )
        alpha_bar = schedule.alpha_bar.to(y.device)[landscape].unsqueeze(-1)
        signal, spread = alpha_bar.sqrt(), (1 - alpha_bar).sqrt()
        context = landscape_context(x, landscape)

        noise = torch.randn(y.shape, generator=generator, **options)
        denoising = denoising_loss(self.model, context, signal * y + spread * noise, noise)
        cost = PassCount(forward=1, backward=1, second_order=1)

        shared = torch.randn(y.shape, generator=generator, **options)
        positive = signal * y + spread * shared
        negative, passes = self._refine(
            context, signal * y + spread * NEGATIVE_NOISE_SCALE * shared, landscape
        )
        negative = signal * (negative / signal).clamp(-CLIP, CLIP) + spread * shared
        real = self.model.energy(context, positive)
        fake = self.model.energy(context, negative)
        contrastive = contrastive_loss(real, fake)
        cost = cost + passes + PassCount(forward=2, backward=2)
        return BatchLoss(denoising + CONTRASTIVE_WEIGHT * contrastive, cost * rows)

    def _refine(
        self, context: Tensor, start: Tensor, landscape: Tensor
    ) -> tuple[Tensor, PassCount]:
        """NEGATIVE_STEPS accept-only-lower steps per row on its own landscape, no graph.

        Rows are grouped by landscape, so every row is clamped to the bound of its own
        landscape. Returns the refined candidates and the passes per row.
        """
        refined = start.detach().clone()
        for k in landscape.unique().tolist():
            rows = landscape == k
            refined[rows], _ = descend(
                self.model, self.steps, context[rows], start[rows].detach(), k, NEGATIVE_STEPS
            )
        return refined, PassCount(forward=2, backward=1) * NEGATIVE_STEPS
