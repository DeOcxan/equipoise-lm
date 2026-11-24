"""Which parameters the earlier steps reach under S2 truncation (method.md section 6.2, U1)."""

import pytest
import torch

from ebt.config import ThinkingConfig, TrainingMode
from ebt.core.objective import trajectory_loss
from support import TanhEnergy, TransformedTanhEnergy, float64_loop

STEPS = 3
ENERGY_PARAMETERS = ("hidden", "bias", "output")
TRANSFORM_PARAMETERS = ("scale", "shift")


def squared_error(prediction, target):
    return ((prediction - target) ** 2).sum()


def gradients(model, config, initial, context, target):
    """Gradients of the S2 loss with respect to every model parameter and α, by name."""
    loop = float64_loop(config)
    trajectory = loop.train_trajectory(model, context, initial)
    loss = trajectory_loss(trajectory.candidates[1:], target, squared_error, config.mode)
    names = [name for name, _ in model.named_parameters()]
    parameters = [parameter for _, parameter in model.named_parameters()]
    if config.step_size_learnable:
        names.append("alpha")
        parameters.append(loop.step_size.raw)
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    result = {}
    for name, parameter, value in zip(names, parameters, values, strict=True):
        result[name] = torch.zeros_like(parameter) if value is None else value
    return result, trajectory


def compare_with_last_step_only(model, energy_scaled=False, learnable_alpha=False):
    """S2 gradients of a k-step run, and of one S2 step started from the run's detached ŷ_{k-1}.

    If the earlier steps reached a parameter, the two gradients for that parameter differ.
    """
    generator = torch.Generator().manual_seed(11)
    context = torch.randn(5, 2, dtype=torch.float64, generator=generator)
    initial = torch.randn(5, 3, dtype=torch.float64, generator=generator)
    target = torch.randn(5, 3, dtype=torch.float64, generator=generator)
    common = {
        "mode": TrainingMode.S2,
        "step_size": 0.3,
        "step_size_learnable": learnable_alpha,
        "energy_scaled_step": energy_scaled,
    }
    full, trajectory = gradients(
        model, ThinkingConfig(**common, steps_min=STEPS, steps_max=STEPS), initial, context, target
    )
    before_last = trajectory.candidates[-2].detach()
    last_only, _ = gradients(
        model, ThinkingConfig(**common, steps_min=1, steps_max=1), before_last, context, target
    )
    return full, last_only


@pytest.mark.parametrize("energy_scaled", [False, True], ids=["fixed-step", "energy-scaled"])
def test_s2_energy_parameters_are_reached_by_the_final_step_only(energy_scaled):
    """Confirms U1 for the energy parameters θ: detaching ŷ_{k-1} leaves dL/dθ unchanged."""
    model = TransformedTanhEnergy(context_dim=2, candidate_dim=3, hidden=5, seed=7).double()
    full, last_only = compare_with_last_step_only(model, energy_scaled=energy_scaled)
    for name in ENERGY_PARAMETERS:
        assert full[name].norm() > 0
        torch.testing.assert_close(full[name], last_only[name], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("energy_scaled", [False, True], ids=["fixed-step", "energy-scaled"])
def test_s2_post_step_transform_is_reached_by_earlier_steps(energy_scaled):
    """Confirms U1 for the post-step transform φ: detaching ŷ_{k-1} changes dL/dφ."""
    model = TransformedTanhEnergy(context_dim=2, candidate_dim=3, hidden=5, seed=7).double()
    full, last_only = compare_with_last_step_only(model, energy_scaled=energy_scaled)
    for name in TRANSFORM_PARAMETERS:
        difference = (full[name] - last_only[name]).norm() / full[name].norm()
        assert difference > 1e-3, name


def test_s2_without_post_step_transform_learns_from_the_final_step_only():
    """With an identity post-step transform and fixed α, earlier steps reach no parameter."""
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=5, seed=7).double()
    full, last_only = compare_with_last_step_only(model)
    for name in ENERGY_PARAMETERS:
        torch.testing.assert_close(full[name], last_only[name], rtol=1e-12, atol=1e-14)


def test_s2_learnable_step_size_would_be_reached_by_earlier_steps():
    """A learnable α is a second path; S2 keeps α fixed (method.md section 10)."""
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=5, seed=7).double()
    full, last_only = compare_with_last_step_only(model, learnable_alpha=True)
    difference = (full["alpha"] - last_only["alpha"]).abs() / full["alpha"].abs()
    assert difference > 1e-3
    for name in ENERGY_PARAMETERS:
        torch.testing.assert_close(full[name], last_only[name], rtol=1e-12, atol=1e-14)
