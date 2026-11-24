"""The thinking loop: gradient descent on the energy (method.md sections 5 to 8 and 10)."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ebt.config import InferenceConfig, ThinkingConfig, TrainingMode
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.core.step_size import StepSize


@dataclass
class Trajectory:
    """Result of one call of the thinking loop."""

    candidates: list[Tensor]
    """ŷ_0, ..., ŷ_k: the initial candidate followed by the candidate after every step."""
    energies: list[Tensor]
    """E(x, v_i) for i < k, with v_i = transform_candidate(ŷ_i + η_i), the thinking variable
    of each step.

    In training, the energy of a step that keeps its second-order graph keeps its graph too, so
    that a loss on it, such as the contrastive loss (method.md section 16), reaches the
    parameters. All other energies, and all energies at inference, are detached."""
    final_energy: Tensor | None
    """E(x, transform_candidate(ŷ_k)), computed only when requested; detached."""
    step_size: Tensor
    """α per energy element after jitter and before energy scaling; detached."""
    steps: int
    cost: PassCount


@dataclass
class Selection:
    """Result of best-of-M selection (method.md section 5.2, Algorithm 2)."""

    best: Tensor
    """Lowest-energy final candidate, chosen separately for every energy element."""
    best_index: Tensor
    """Index in 0..M-1 of the chosen candidate, shape energy_shape."""
    candidates: Tensor
    """All final candidates, shape (M, *candidate_shape)."""
    final_energies: Tensor
    """Final energies of all candidates, shape (M, *energy_shape)."""
    cost: PassCount
    """Passes per input example, M times the passes of one candidate."""


class ThinkingLoop(nn.Module):
    """Runs ŷ ← after_step(v − α ∇_v E(x, v)), v = transform_candidate(ŷ + η), and owns α.

    For an identity transform and no noise this is ŷ ← ŷ − α ∇ŷ E(x, ŷ) (method.md sections
    4.1, 5 and 10).
    """

    def __init__(self, config: ThinkingConfig) -> None:
        super().__init__()
        self.config = config
        self.step_size = StepSize(
            config.step_size,
            learnable=config.step_size_learnable,
            jitter=config.step_size_jitter,
            energy_scaled=config.energy_scaled_step,
        )

    def train_trajectory(
        self,
        model: EnergyModel,
        context: Tensor | None,
        initial: Tensor,
        generator: torch.Generator | None = None,
    ) -> Trajectory:
        """Thinking during training (method.md sections 6.2, 7, 8.1, 8.2).

        The step count is drawn once per call from {steps_min, ..., steps_max}; α is jittered
        per energy element; Langevin noise uses `config.langevin_std`. The graph kept between
        steps follows `config.mode` as in the table of method.md section 6.2. The returned
        candidates keep their graph so that the training loss reaches the parameters.
        """
        steps = self._sample_steps(generator)
        return self._run(
            model,
            context,
            initial,
            steps=steps,
            langevin_std=self.config.langevin_std,
            training=True,
            generator=generator,
            final_energy=False,
        )

    def infer(
        self,
        model: EnergyModel,
        context: Tensor | None,
        initial: Tensor,
        config: InferenceConfig,
        generator: torch.Generator | None = None,
        final_energy: bool = False,
    ) -> Trajectory:
        """Thinking at evaluation (method.md sections 5.1 and 7).

        Runs exactly `config.steps` steps with the clamped α, no step-size or step-count
        randomness and Langevin noise `config.langevin_std`. Every step is detached and no
        second-order graph is built, whatever the training mode. Works inside torch.no_grad():
        gradients are enabled internally for the candidate only. All outputs are detached.
        """
        return self._run(
            model,
            context,
            initial,
            steps=config.steps,
            langevin_std=config.langevin_std,
            training=False,
            generator=generator,
            final_energy=final_energy,
        )

    def best_of(
        self,
        model: EnergyModel,
        context: Tensor | None,
        initial: Tensor,
        config: InferenceConfig,
        generator: torch.Generator | None = None,
    ) -> Selection:
        """Best-of-M selection by lowest final energy (method.md section 5.2, Algorithm 2).

        `initial` has shape (M, *candidate_shape) with M = `config.samples`. The M candidates
        are optimized independently with `infer`, and for every energy element (every example,
        or every position for sequences) the candidate with the lowest final energy is chosen.
        """
        samples = config.samples
        if initial.shape[0] != samples:
            raise ValueError(f"initial has {initial.shape[0]} candidates, expected {samples}")
        candidate_shape = initial.shape[1:]
        flat_initial = initial.reshape(samples * candidate_shape[0], *candidate_shape[1:])
        flat_context = None
        if context is not None:
            flat_context = context.repeat(samples, *([1] * (context.dim() - 1)))

        trajectory = self.infer(
            model, flat_context, flat_initial, config, generator, final_energy=True
        )
        candidates = trajectory.candidates[-1].reshape(samples, *candidate_shape)
        energy_shape = trajectory.final_energy.shape
        final_energies = trajectory.final_energy.reshape(
            samples, candidate_shape[0], *energy_shape[1:]
        )

        best_index = final_energies.argmin(dim=0)
        event_dims = len(candidate_shape) - best_index.dim()
        gather_index = best_index.reshape(1, *best_index.shape, *([1] * event_dims))
        best = torch.take_along_dim(candidates, gather_index.expand(1, *candidate_shape), dim=0)
        return Selection(
            best=best[0],
            best_index=best_index,
            candidates=candidates,
            final_energies=final_energies,
            cost=trajectory.cost * samples,
        )

    def _sample_steps(self, generator: torch.Generator | None) -> int:
        """Step count for one training call, uniform in {steps_min, ..., steps_max} (8.2)."""
        low, high = self.config.steps_min, self.config.steps_max
        if low == high:
            return low
        device = generator.device if generator is not None else "cpu"
        return int(torch.randint(low, high + 1, (1,), generator=generator, device=device).item())

    def _run(
        self,
        model: EnergyModel,
        context: Tensor | None,
        initial: Tensor,
        *,
        steps: int,
        langevin_std: float,
        training: bool,
        generator: torch.Generator | None,
        final_energy: bool,
    ) -> Trajectory:
        mode = self.config.mode
        energy_shape = model.energy_shape(initial)
        cost = PassCount()

        with torch.enable_grad():
            alpha = self.step_size.sample(energy_shape, training, generator).to(initial.dtype)
            candidate = initial
            candidates = [initial.detach() if not training else initial]
            energies = []
            for i in range(steps):
                keep_graph = training and (mode is not TrainingMode.S2 or i == steps - 1)

                # S1 and evaluation start every step from a detached candidate. S2 and full
                # unroll keep the chain; a candidate without a graph becomes a fresh leaf.
                if not training or mode is TrainingMode.S1 or not candidate.requires_grad:
                    candidate = candidate.detach().requires_grad_(True)

                # Langevin noise is added before the energy (method.md section 7, disagreement
                # D4 with Eq. 2 of the paper). The gradient is taken with respect to the
                # thinking variable of the noisy point, and the step starts from it (4.1).
                noisy = candidate
                if langevin_std > 0:
                    noisy = candidate + langevin_std * torch.randn(
                        candidate.shape,
                        generator=generator,
                        dtype=candidate.dtype,
                        device=candidate.device,
                    )

                variable = model.transform_candidate(noisy)
                energy = self._energy(model, context, variable, energy_shape)
                cost.forward += 1
                (gradient,) = torch.autograd.grad(energy.sum(), variable, create_graph=keep_graph)
                cost.backward += 1
                if keep_graph:
                    cost.second_order += 1
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("non-finite candidate gradient during thinking")

                # Under truncation and at evaluation the energy in the step size is a constant.
                step_energy = energy if keep_graph else energy.detach()
                step = self.step_size.scale(alpha, step_energy)
                step = step.reshape(*step.shape, *([1] * (variable.dim() - step.dim())))
                if training:
                    candidate = model.after_step(variable - step * gradient)
                else:
                    with torch.no_grad():
                        candidate = model.after_step(variable - step * gradient)
                candidates.append(candidate if training else candidate.detach())
                energies.append(energy if keep_graph else energy.detach())

            last_energy = None
            if final_energy:
                with torch.no_grad():
                    variable = model.transform_candidate(candidate.detach())
                    last_energy = self._energy(model, context, variable, energy_shape)
                cost.forward += 1

        return Trajectory(
            candidates=candidates,
            energies=energies,
            final_energy=last_energy,
            step_size=alpha.detach(),
            steps=steps,
            cost=cost,
        )

    @staticmethod
    def _energy(
        model: EnergyModel, context: Tensor | None, variable: Tensor, energy_shape: torch.Size
    ) -> Tensor:
        energy = model.energy(context, model.encode_candidate(variable))
        if energy.shape != energy_shape:
            raise ValueError(
                f"energy has shape {tuple(energy.shape)}, expected {tuple(energy_shape)}"
            )
        return energy
