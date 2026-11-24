"""Pass counts per input example (method.md section 6.3)."""

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig, TrainingMode
from ebt.core.cost import PassCount
from support import TanhEnergy, float64_loop

TRAINING_COST = {
    # mode: (forward, backward, second_order) for k steps
    TrainingMode.S1: lambda k: (k, k, k),
    TrainingMode.S2: lambda k: (k, k, 1),
    TrainingMode.FULL: lambda k: (k, k, k),
}


def setup():
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=4, seed=12).double()
    context = torch.zeros(4, 2, dtype=torch.float64)
    initial = torch.ones(4, 3, dtype=torch.float64)
    return model, context, initial


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_training_pass_counts(mode, steps):
    model, context, initial = setup()
    loop = float64_loop(ThinkingConfig(mode=mode, step_size=0.1, steps_min=steps, steps_max=steps))
    trajectory = loop.train_trajectory(model, context, initial)
    assert trajectory.cost == PassCount(*TRAINING_COST[mode](steps))


@pytest.mark.parametrize("final_energy", [False, True])
def test_inference_pass_counts(final_energy):
    model, context, initial = setup()
    loop = float64_loop(ThinkingConfig(mode=TrainingMode.FULL, step_size=0.1))
    trajectory = loop.infer(
        model, context, initial, InferenceConfig(steps=3), final_energy=final_energy
    )
    assert trajectory.cost == PassCount(3 + int(final_energy), 3, 0)


def test_best_of_pass_counts_scale_with_the_number_of_samples():
    model, context, initial = setup()
    loop = float64_loop(ThinkingConfig(step_size=0.1))
    candidates = initial.expand(3, *initial.shape).clone()
    selection = loop.best_of(model, context, candidates, InferenceConfig(steps=2, samples=3))
    assert selection.cost == PassCount(3 * (2 + 1), 3 * 2, 0)
