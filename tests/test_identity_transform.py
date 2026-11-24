"""With the identity candidate transform, ThinkingLoop equals the direct noisy-candidate update.

The regression and image EBTs use the identity as their candidate transform, so the thinking
variable is the noisy candidate itself (method.md section 4.1). `ReferenceLoop` writes that update
directly, y <- after_step((y + η) - α ∇E(y + η)), with the gradient taken at the noisy candidate
and the same graph and random-number use. Both loops run on the same inputs and generators;
losses, parameter gradients and inference trajectories must be bitwise equal. The comparison is
made on the machine that runs the test, so it does not depend on platform-specific floating-point
results (docs/numerics.md).
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from ebt.config import InferenceConfig, ObjectiveConfig, Tier, TrainingMode
from ebt.core.cost import PassCount
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop, Trajectory
from ebt.experiments import mnist_denoising, regression_g2
from ebt.models.mlp import MLPEnergy
from ebt.models.patch_transformer import PatchEnergy
from ebt.training.batch import Batch
from ebt.training.objectives import EBTObjective


class ReferenceLoop(ThinkingLoop):
    """The thinking step written directly on the noisy candidate, with the gradient taken there."""

    def _run(
        self, model, context, initial, *, steps, langevin_std, training, generator, final_energy
    ) -> Trajectory:
        shape = model.energy_shape(initial)
        cost = PassCount()
        with torch.enable_grad():
            alpha = self.step_size.sample(shape, training, generator).to(initial.dtype)
            y = initial
            candidates, energies = [initial if training else initial.detach()], []
            for i in range(steps):
                second_order = training and (
                    self.config.mode is not TrainingMode.S2 or i == steps - 1
                )
                if not training or self.config.mode is TrainingMode.S1 or not y.requires_grad:
                    y = y.detach().requires_grad_(True)
                if langevin_std > 0:
                    noise = torch.randn(
                        y.shape, generator=generator, dtype=y.dtype, device=y.device
                    )
                    y = y + langevin_std * noise
                energy = model.energy(context, model.encode_candidate(y))
                (gradient,) = torch.autograd.grad(energy.sum(), y, create_graph=second_order)
                cost.forward += 1
                cost.backward += 1
                cost.second_order += int(second_order)
                scaled = self.step_size.scale(alpha, energy if second_order else energy.detach())
                scaled = scaled.reshape(*scaled.shape, *([1] * (y.dim() - scaled.dim())))
                with torch.set_grad_enabled(training):
                    y = model.after_step(y - scaled * gradient)
                candidates.append(y if training else y.detach())
                energies.append(energy if second_order else energy.detach())
            last = None
            if final_energy:
                with torch.no_grad():
                    last = model.energy(context, model.encode_candidate(y.detach()))
                cost.forward += 1
        return Trajectory(candidates, energies, last, alpha.detach(), steps, cost)


REGRESSION_CASES = [
    ("ebt_mse", None),
    ("ebt_mse", TrainingMode.S1),
    ("ebt_mse", TrainingMode.FULL),
    ("ebt_contrastive", None),
    ("ebt_combined", None),
    ("ebt_regularized", None),
]
BATCHES = 3


def train_and_infer(objective, loop, batches, inference, inference_batch, initial):
    """Every loss, every parameter gradient and one inference trajectory, detached."""
    out = {}
    generator = torch.Generator().manual_seed(7)
    for b, batch in enumerate(batches):
        objective.module.zero_grad(set_to_none=True)
        loss = objective.batch_loss(batch, generator).loss
        loss.backward()
        out[f"loss{b}"] = loss.detach()
        for name, parameter in objective.module.named_parameters():
            if parameter.grad is not None:
                out[f"grad{b}.{name}"] = parameter.grad.clone()
    trajectory = loop.infer(
        objective.model,
        inference_batch.context,
        initial,
        inference,
        torch.Generator().manual_seed(8),
        final_energy=True,
    )
    for i, candidate in enumerate(trajectory.candidates):
        out[f"infer.candidate{i}"] = candidate
    for i, energy in enumerate(trajectory.energies):
        out[f"infer.energy{i}"] = energy
    out["infer.final_energy"] = trajectory.final_energy
    return out


def regression_case(loop_class, model: str, mode: TrainingMode | None) -> dict[str, Tensor]:
    torch.manual_seed(0)
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    thinking = regression_g2.thinking_config(candidate)
    if mode is not None:
        thinking = dataclasses.replace(thinking, mode=mode)
    energy = MLPEnergy(1, 1, regression_g2.ENERGY_MODEL)
    loop = loop_class(thinking)
    objective = EBTObjective(
        energy,
        loop,
        F.mse_loss,
        regression_g2.objective_weights(candidate),
        initial=lambda batch, g: torch.randn(batch.target.shape, generator=g),
        replay=ReplayBuffer(regression_g2.REPLAY, torch.Generator().manual_seed(3)),
    )
    data = torch.Generator().manual_seed(1)
    batches = []
    for _ in range(BATCHES):
        x = 4 * torch.rand(32, 1, generator=data)
        sign = torch.randint(0, 2, (32, 1), generator=data) * 2 - 1
        batches.append(Batch(x, sign * x.sqrt()))
    inference = InferenceConfig(steps=4, langevin_std=0.1)
    initial = torch.randn(32, 1, generator=data)
    return train_and_infer(objective, loop, batches, inference, batches[0], initial)


def image_case(loop_class) -> dict[str, Tensor]:
    torch.manual_seed(0)
    settings = mnist_denoising.TIERS[Tier.FAST]
    candidate = mnist_denoising.search_space("ebt", Tier.FAST)[0]
    energy = PatchEnergy(settings.model)
    # The zero-initialized output and modulation layers would make most parameter gradients
    # exactly zero in both loops; seeded offsets make every gradient informative.
    offsets = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in energy.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=offsets))
    loop = loop_class(mnist_denoising.thinking_config(candidate))
    data = torch.Generator().manual_seed(1)

    def noised(batch, g):
        return batch.target + 0.3 * torch.randn(batch.target.shape, generator=g)

    objective = EBTObjective(energy, loop, mnist_denoising.smooth_l1, ObjectiveConfig(), noised)
    batches = [
        Batch(None, 2 * torch.rand(4, 1, 28, 28, generator=data) - 1) for _ in range(BATCHES)
    ]
    initial = noised(batches[0], data)
    return train_and_infer(objective, loop, batches, InferenceConfig(steps=2), batches[0], initial)


CASES = {
    **{
        f"{model}-{mode.value if mode else 'default'}": (
            lambda loop_class, model=model, mode=mode: regression_case(loop_class, model, mode)
        )
        for model, mode in REGRESSION_CASES
    },
    "mnist_ebt": image_case,
}


@pytest.mark.parametrize("name", CASES)
def test_identity_transform_equals_reference_loop_bitwise(name):
    reference = CASES[name](ReferenceLoop)
    current = CASES[name](ThinkingLoop)
    assert current.keys() == reference.keys()
    assert [key for key in reference if not torch.equal(current[key], reference[key])] == []
