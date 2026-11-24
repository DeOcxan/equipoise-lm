"""Which recorded energies keep their graph (method.md sections 6.2 and 16.4)."""

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig, TrainingMode
from support import TanhEnergy, float64_loop

STEPS = 3
GRAPH_KEPT = {
    TrainingMode.S1: [True, True, True],
    TrainingMode.S2: [False, False, True],
    TrainingMode.FULL: [True, True, True],
}


def setup(mode):
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=4, seed=20).double()
    generator = torch.Generator().manual_seed(21)
    context = torch.randn(4, 2, dtype=torch.float64, generator=generator)
    initial = torch.randn(4, 3, dtype=torch.float64, generator=generator)
    loop = float64_loop(ThinkingConfig(mode=mode, step_size=0.2, steps_min=STEPS, steps_max=STEPS))
    return model, context, initial, loop


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_training_energies_keep_their_graph_exactly_on_second_order_steps(mode):
    model, context, initial, loop = setup(mode)
    trajectory = loop.train_trajectory(model, context, initial)
    assert [energy.requires_grad for energy in trajectory.energies] == GRAPH_KEPT[mode]
    (gradient,) = torch.autograd.grad(trajectory.energies[-1].sum(), model.output)
    assert gradient.norm() > 0


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_inference_energies_are_detached(mode):
    model, context, initial, loop = setup(mode)
    trajectory = loop.infer(
        model, context, initial, InferenceConfig(steps=STEPS), final_energy=True
    )
    assert not any(energy.requires_grad for energy in trajectory.energies)
    assert not trajectory.final_energy.requires_grad
