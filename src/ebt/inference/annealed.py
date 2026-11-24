"""Annealed optimization over IRED's energy landscapes (method.md section 16.8).

Follows the procedure of the authors' code (`diffusion_lib/denoising_diffusion_pytorch_1d.py`:
`p_sample_loop`, `p_sample` and `opt_step`) for continuous tasks, described in our own words.
Every gradient step is one call of `ThinkingLoop.infer` with one step; the acceptance test,
the clamping and the rescaling between landscapes are done here, outside the loop.
"""

import torch
from torch import Tensor, nn

from ebt.config import InferenceConfig, ThinkingConfig
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.core.thinking import ThinkingLoop
from ebt.models.ired import LandscapeSchedule, landscape_context

CLIP = 2.0
"""Bound of the predicted clean value for continuous tasks; the clamp on landscape k is
CLIP · √ᾱ_k (the authors' scale factor 2 for continuous tasks)."""
LATER_STEP_FACTOR = 2.0
"""Factor on λ_k for every descent step after the first on a landscape. In the authors'
`opt_step` the step factor starts at 1 and is reassigned to the continuous clamp factor 2
after the first step (method.md section 16.8)."""
DESCENT_STEPS = 5
"""Accept-only-lower steps per landscape at inference (the authors' value outside Sudoku)."""
SINGLE_STEP = InferenceConfig(steps=1)


def _loop(step_size: float) -> ThinkingLoop:
    return ThinkingLoop(ThinkingConfig(step_size=step_size))


class LandscapeSteps(nn.Module):
    """The fixed-α thinking loops of every landscape k.

    - `jump[k]`: α = √(1 − ᾱ_k), the step that predicts the clean value;
    - `first[k]`: α = λ_k, the first descent step;
    - `later[k]`: α = 2 λ_k, every later descent step.
    """

    def __init__(self, schedule: LandscapeSchedule) -> None:
        super().__init__()
        self.schedule = schedule
        alpha_bar = schedule.alpha_bar.tolist()
        step = schedule.step_size.tolist()
        self.jump = nn.ModuleList(_loop((1 - a) ** 0.5) for a in alpha_bar)
        self.first = nn.ModuleList(_loop(s) for s in step)
        self.later = nn.ModuleList(_loop(LATER_STEP_FACTOR * s) for s in step)

    def bound(self, landscape: int) -> float:
        """Clamp bound CLIP · √ᾱ_k of the candidate on landscape k."""
        return CLIP * self.schedule.alpha_bar[landscape].item() ** 0.5


def accept_lower_step(
    model: EnergyModel, loop: ThinkingLoop, context: Tensor, candidate: Tensor, bound: float
) -> tuple[Tensor, PassCount]:
    """One gradient step, clamped to ±bound, kept only where it does not raise the energy.

    Rows whose clamped proposal has a higher energy than their current candidate keep the
    current candidate (the authors' `opt_step`). Costs two forward and one backward pass.
    """
    trajectory = loop.infer(model, context, candidate, SINGLE_STEP)
    proposal = trajectory.candidates[-1].clamp(-bound, bound)
    with torch.no_grad():
        proposal_energy = model.energy(context, model.encode_candidate(proposal))
    worse = proposal_energy > trajectory.energies[0]
    worse = worse.reshape(*worse.shape, *([1] * (candidate.dim() - worse.dim())))
    return torch.where(worse, candidate.detach(), proposal), trajectory.cost + PassCount(forward=1)


def descend(
    model: EnergyModel,
    steps: LandscapeSteps,
    context: Tensor,
    candidate: Tensor,
    landscape: int,
    count: int,
) -> tuple[Tensor, PassCount]:
    """`count` accept-only-lower steps on one landscape, all rows on that landscape."""
    cost = PassCount()
    for i in range(count):
        loop = steps.first[landscape] if i == 0 else steps.later[landscape]
        candidate, passes = accept_lower_step(
            model, loop, context, candidate, steps.bound(landscape)
        )
        cost = cost + passes
    return candidate, cost


def annealed_sample(
    model: EnergyModel, steps: LandscapeSteps, x: Tensor, initial: Tensor
) -> tuple[Tensor, PassCount]:
    """IRED inference from `initial` (the authors' `p_sample_loop`); returns (B, D) and passes.

    For each landscape k from K − 1 down to 0:

    1. Jump: predict the clean value ŷ_0 = (ŷ − √(1 − ᾱ_k) ∇E(x, ŷ; k)) / √ᾱ_k, clamp it to
       ±CLIP and set ŷ = √ᾱ_k · ŷ_0.
    2. Descend: DESCENT_STEPS accept-only-lower steps, each clamped to ±CLIP · √ᾱ_k.
    3. Clamp ŷ to ±CLIP · √ᾱ_k and, unless k = 0, rescale it to the next landscape:
       ŷ ← √ᾱ_{k−1} / √ᾱ_k · ŷ.

    The result is the candidate after landscape 0, which still carries the factor √ᾱ_0, as
    in the authors' code.
    """
    alpha_bar = steps.schedule.alpha_bar.tolist()
    candidate, cost = initial, PassCount()
    for landscape in reversed(range(steps.schedule.landscapes)):
        context = landscape_context(x, landscape)
        scale = alpha_bar[landscape] ** 0.5
        trajectory = steps.jump[landscape].infer(model, context, candidate, SINGLE_STEP)
        candidate = scale * (trajectory.candidates[-1] / scale).clamp(-CLIP, CLIP)
        cost = cost + trajectory.cost
        candidate, passes = descend(model, steps, context, candidate, landscape, DESCENT_STEPS)
        cost = cost + passes
        bound = steps.bound(landscape)
        candidate = candidate.clamp(-bound, bound)
        if landscape > 0:
            candidate = candidate * (alpha_bar[landscape - 1] ** 0.5 / scale)
    return candidate, cost
