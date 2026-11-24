"""Step-size jitter, random step count and Langevin noise (method.md sections 7, 8)."""

from collections import Counter

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig
from ebt.core.analytic import QuadraticEnergy
from ebt.core.step_size import MIN_STEP_SIZE, StepSize
from support import TanhEnergy, float64_loop


@pytest.mark.parametrize("energy_shape", [(64,), (16, 12)], ids=["per-example", "per-position"])
def test_jittered_step_size_lies_in_range_with_energy_shape(energy_shape):
    step_size = StepSize(2.0, learnable=False, jitter=3.0, energy_scaled=False)
    generator = torch.Generator().manual_seed(0)
    alpha = step_size.sample(torch.Size(energy_shape), training=True, generator=generator)
    assert alpha.shape == energy_shape
    assert (alpha >= 2.0 / 3.0).all() and (alpha <= 2.0 * 3.0).all()
    assert alpha.unique().numel() > 1


def test_step_size_is_fixed_and_clamped_outside_training():
    step_size = StepSize(2.0, learnable=False, jitter=3.0, energy_scaled=False)
    alpha = step_size.sample(torch.Size((5,)), training=False, generator=None)
    assert (alpha == 2.0).all()
    tiny = StepSize(1e-8, learnable=False, jitter=1.0, energy_scaled=False)
    assert (tiny.sample(torch.Size((2,)), training=True, generator=None) == MIN_STEP_SIZE).all()


def test_training_trajectory_records_the_step_size_per_energy_element():
    model = TanhEnergy(context_dim=1, candidate_dim=2, hidden=3, seed=0).double()
    loop = float64_loop(ThinkingConfig(step_size=0.1, step_size_jitter=2.0))
    context = torch.zeros(6, 4, 1, dtype=torch.float64)
    initial = torch.zeros(6, 4, 2, dtype=torch.float64)
    trajectory = loop.train_trajectory(model, context, initial, torch.Generator().manual_seed(1))
    assert trajectory.step_size.shape == (6, 4)
    assert (trajectory.step_size >= 0.05).all() and (trajectory.step_size <= 0.2).all()


def test_step_count_is_uniform_over_the_configured_range():
    """k is drawn from {steps_min, ..., steps_max} once per call (method.md section 8.2).

    With steps_min = 2 and steps_max = 4 this is the upstream text S2 range {2, 3, 4}.
    """
    model = QuadraticEnergy(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    loop = float64_loop(ThinkingConfig(step_size=0.1, steps_min=2, steps_max=4))
    generator = torch.Generator().manual_seed(2)
    counts = Counter()
    for _ in range(600):
        trajectory = loop.train_trajectory(
            model, None, torch.ones(1, 2, dtype=torch.float64), generator
        )
        assert len(trajectory.candidates) == trajectory.steps + 1
        assert len(trajectory.energies) == trajectory.steps
        counts[trajectory.steps] += 1
    assert set(counts) == {2, 3, 4}
    for value in (2, 3, 4):
        assert abs(counts[value] / 600 - 1 / 3) < 0.06, counts


def test_inference_uses_the_given_step_count():
    model = QuadraticEnergy(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    loop = float64_loop(ThinkingConfig(step_size=0.1, steps_min=2, steps_max=4))
    trajectory = loop.infer(
        model, None, torch.ones(1, 2, dtype=torch.float64), InferenceConfig(steps=7)
    )
    assert trajectory.steps == 7 and len(trajectory.candidates) == 8


def test_langevin_noise_has_the_configured_standard_deviation():
    """With a flat energy the step is ŷ_1 = ŷ_0 + η, so ŷ_1 - ŷ_0 is the noise itself."""
    flat = QuadraticEnergy(
        torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    loop = float64_loop(ThinkingConfig(step_size=0.1, langevin_std=0.7))
    initial = torch.zeros(100_000, 1, dtype=torch.float64)
    trajectory = loop.train_trajectory(flat, None, initial, torch.Generator().manual_seed(3))
    noise = trajectory.candidates[1] - initial
    assert abs(noise.std().item() / 0.7 - 1) < 0.02
    assert abs(noise.mean().item()) < 0.01


def test_gradient_is_taken_at_the_noisy_point():
    """The update is ŷ_1 = (ŷ_0 + η) - α∇E(ŷ_0 + η), not ŷ_0 + η - α∇E(ŷ_0) (method.md 7, D4)."""
    precision = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    center = torch.tensor([1.0, -1.0], dtype=torch.float64)
    model = QuadraticEnergy(precision, center)
    loop = float64_loop(ThinkingConfig(step_size=0.1, langevin_std=0.5))
    initial = torch.tensor([[0.3, 0.4], [-0.2, 0.9]], dtype=torch.float64)

    generator = torch.Generator().manual_seed(4)
    expected_noise = torch.randn(
        initial.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64
    )
    trajectory = loop.train_trajectory(model, None, initial, generator)

    noisy = initial + 0.5 * expected_noise
    at_noisy_point = noisy - 0.1 * (noisy - center) @ precision
    at_clean_point = noisy - 0.1 * (initial - center) @ precision
    torch.testing.assert_close(trajectory.candidates[1], at_noisy_point, rtol=0, atol=1e-14)
    assert (trajectory.candidates[1] - at_clean_point).abs().max() > 1e-3


def test_evaluation_uses_the_inference_noise_not_the_training_noise():
    model = QuadraticEnergy(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    loop = float64_loop(ThinkingConfig(step_size=0.1, langevin_std=3.0))
    initial = torch.ones(3, 2, dtype=torch.float64)
    trajectory = loop.infer(model, None, initial, InferenceConfig(steps=1))
    torch.testing.assert_close(trajectory.candidates[1], 0.9 * initial, rtol=0, atol=1e-15)


def test_runs_without_randomness_are_identical():
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=4, seed=5).double()
    context = torch.randn(4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    initial = torch.randn(4, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    loop = float64_loop(ThinkingConfig(step_size=0.2, steps_min=3, steps_max=3))
    first = loop.train_trajectory(model, context, initial)
    second = loop.train_trajectory(model, context, initial)
    assert torch.equal(first.candidates[-1], second.candidates[-1])
    inference = InferenceConfig(steps=3)
    assert torch.equal(
        loop.infer(model, context, initial, inference).candidates[-1],
        loop.infer(model, context, initial, inference).candidates[-1],
    )
