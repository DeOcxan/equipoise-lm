"""Gradients of the thinking loop against finite differences and a hand derivation (G1)."""

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig, TrainingMode
from ebt.core.analytic import QuadraticEnergy
from ebt.core.objective import trajectory_loss
from ebt.core.step_size import ENERGY_SCALE_TEMPERATURE
from support import TanhEnergy, TransformedTanhEnergy, float64_loop

STEP = 1e-6
"""Central-difference step in float64."""
TOLERANCE = 1e-6
"""Maximum relative error against finite differences (questions.md, G1)."""


def squared_error(prediction, target):
    return ((prediction - target) ** 2).sum()


def relative_error(computed, reference):
    return ((computed - reference).norm() / reference.norm()).item()


def central_difference(function, tensor):
    """Central differences of a scalar `function()` with respect to every element of `tensor`."""
    gradient = torch.zeros_like(tensor)
    flat, flat_gradient = tensor.data.view(-1), gradient.view(-1)
    for index in range(flat.numel()):
        original = flat[index].item()
        flat[index] = original + STEP
        upper = function()
        flat[index] = original - STEP
        lower = function()
        flat[index] = original
        flat_gradient[index] = (upper - lower) / (2 * STEP)
    return gradient


@pytest.mark.parametrize("sequence", [False, True], ids=["per-example", "per-position"])
def test_candidate_gradient_matches_finite_differences(sequence):
    leading = (3, 4) if sequence else (3,)
    model = TanhEnergy(context_dim=2, candidate_dim=5, hidden=7, seed=0).double()
    context = torch.randn(
        *leading, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
    )
    initial = torch.randn(
        *leading, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
    )
    loop = float64_loop(ThinkingConfig(step_size=0.5))

    trajectory = loop.infer(model, context, initial, InferenceConfig(steps=1))
    loop_gradient = (initial - trajectory.candidates[1]) / 0.5

    probe = initial.clone()
    with torch.no_grad():
        reference = central_difference(lambda: model.energy(context, probe).sum().item(), probe)
    assert relative_error(loop_gradient, reference) <= TOLERANCE


def defining_loss(model, alpha, context, initial, target, mode, steps, energy_scaled, held):
    """Value of the function whose gradient each mode defines (method.md section 6.2).

    The modes differ in which quantities are constants with respect to the parameters:
    S1 fixes the input of every step (the detach); S2 fixes the candidate gradient and the
    energy of every non-final step (the truncation); full unroll fixes nothing. `held` holds
    those constants, recorded at the base parameters; with `held` empty they are recorded.
    """
    candidate, losses = initial, []
    for i in range(steps):
        last = i == steps - 1
        if mode is TrainingMode.S1:
            if len(held["inputs"]) <= i:
                held["inputs"].append(candidate.detach().clone())
            candidate = held["inputs"][i]
        if mode is TrainingMode.S2 and not last:
            if len(held["gradients"]) <= i:
                gradient, energy = energy_and_gradient(model, context, candidate)
                held["gradients"].append(gradient)
                held["energies"].append(energy)
            gradient, energy = held["gradients"][i], held["energies"][i]
        else:
            gradient, energy = energy_and_gradient(model, context, candidate)
        step = alpha * torch.exp(energy / ENERGY_SCALE_TEMPERATURE) if energy_scaled else alpha
        candidate = model.after_step(candidate - step.unsqueeze(-1) * gradient)
        losses.append(squared_error(candidate, target))
    loss = torch.stack(losses).mean() if mode is TrainingMode.S1 else losses[-1]
    return loss.item()


def energy_and_gradient(model, context, candidate):
    with torch.enable_grad():
        point = candidate.detach().requires_grad_(True)
        energy = model.energy(context, point)
        (gradient,) = torch.autograd.grad(energy.sum(), point)
    return gradient.detach(), energy.detach()


@pytest.mark.parametrize("energy_scaled", [False, True], ids=["fixed-step", "energy-scaled"])
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_parameter_gradient_matches_finite_differences(mode, energy_scaled):
    """Autograd through the loop equals central differences of the mode's defining function.

    Covers the energy parameters, the post-step transform parameters and a learnable α.
    """
    steps = 3
    model = TransformedTanhEnergy(context_dim=2, candidate_dim=3, hidden=5, seed=3).double()
    generator = torch.Generator().manual_seed(4)
    context = torch.randn(4, 2, dtype=torch.float64, generator=generator)
    initial = torch.randn(4, 3, dtype=torch.float64, generator=generator)
    target = torch.randn(4, 3, dtype=torch.float64, generator=generator)
    config = ThinkingConfig(
        mode=mode,
        step_size=0.3,
        step_size_learnable=True,
        energy_scaled_step=energy_scaled,
        steps_min=steps,
        steps_max=steps,
    )
    loop = float64_loop(config)

    trajectory = loop.train_trajectory(model, context, initial)
    loss = trajectory_loss(trajectory.candidates[1:], target, squared_error, mode)
    parameters = [*model.parameters(), loop.step_size.raw]
    computed = torch.autograd.grad(loss, parameters)

    held = {"inputs": [], "gradients": [], "energies": []}

    def function():
        alpha = loop.step_size.raw.expand(4)
        return defining_loss(
            model, alpha, context, initial, target, mode, steps, energy_scaled, held
        )

    with torch.no_grad():
        function()  # records the constants at the base parameters
        for parameter, gradient in zip(parameters, computed, strict=True):
            reference = central_difference(function, parameter)
            assert reference.norm() > 0
            assert relative_error(gradient, reference) <= TOLERANCE


def test_one_dimensional_quadratic_matches_hand_derivation():
    """Parameter gradients of the three modes on E(y) = (w/2)(y - c)**2, derived by hand.

    Setup: y_0 = 0, w = 1.5, c = 2, α = 0.2, two steps, loss J(ŷ) = (ŷ - t)**2 with t = 1.5.

    One step: g = ∂E/∂y = w(y - c), so y' = y - αw(y - c) = q·y + a·c with a = αw = 0.3 and
    q = 1 - a = 0.7. Hence y_1 = q·y_0 + a·c = 0.6 and y_2 = q·y_1 + a·c = 0.42 + 0.6 = 1.02.
    Derivatives of one step with its input y held fixed: ∂y'/∂w = -α(y - c), ∂y'/∂c = a.
    Residuals: y_1 - t = -0.9, y_2 - t = -0.48; offsets: y_0 - c = -2, y_1 - c = -1.4.

    Full unroll (no detach, graph at every step, loss on y_2):
        y_2 = q**2·y_0 + (1 - q**2)·c,
        so ∂y_2/∂w = -2αq(y_0 - c) = 0.56 and ∂y_2/∂c = 1 - q**2 = 0.51.
        dJ/dw = 2(y_2 - t)·0.56 = -0.5376
        dJ/dc = 2(y_2 - t)·0.51 = -0.4896

    S1 (detach between steps, loss averaged over both steps):
        L = ½[(y_1 - t)**2 + (y_2 - t)**2], with y_1 treated as a constant input of step 2.
        dL/dw = ½[2(y_1 - t)(-α(y_0 - c)) + 2(y_2 - t)(-α(y_1 - c))]
              = -α[(y_1 - t)(y_0 - c) + (y_2 - t)(y_1 - c)] = -0.2·(1.8 + 0.672) = -0.4944
        dL/dc = a[(y_1 - t) + (y_2 - t)] = 0.3·(-1.38) = -0.414

    S2 (no detach, graph at the final step only, loss on y_2):
        The step-1 gradient g_0 = w(y_0 - c) is a constant, so y_1 = y_0 - αg_0 does not depend on
        (w, c); only step 2 contributes.
        dJ/dw = 2(y_2 - t)(-α(y_1 - c)) = 2·(-0.48)·0.28 = -0.2688
        dJ/dc = 2(y_2 - t)·a = 2·(-0.48)·0.3 = -0.288
    """
    expected = {
        TrainingMode.FULL: (-0.5376, -0.4896),
        TrainingMode.S1: (-0.4944, -0.414),
        TrainingMode.S2: (-0.2688, -0.288),
    }
    for mode, (expected_w, expected_c) in expected.items():
        model = QuadraticEnergy(
            torch.tensor([[1.5]], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)
        )
        loop = float64_loop(ThinkingConfig(mode=mode, step_size=0.2, steps_min=2, steps_max=2))
        initial = torch.zeros(1, 1, dtype=torch.float64)
        target = torch.full((1, 1), 1.5, dtype=torch.float64)

        trajectory = loop.train_trajectory(model, None, initial)
        assert trajectory.candidates[1].item() == pytest.approx(0.6, abs=1e-12)
        assert trajectory.candidates[2].item() == pytest.approx(1.02, abs=1e-12)
        loss = trajectory_loss(trajectory.candidates[1:], target, squared_error, mode)
        grad_w, grad_c = torch.autograd.grad(loss, [model.precision, model.center])
        assert grad_w.item() == pytest.approx(expected_w, abs=1e-12), mode
        assert grad_c.item() == pytest.approx(expected_c, abs=1e-12), mode
