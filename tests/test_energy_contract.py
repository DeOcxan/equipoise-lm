"""The EnergyModel contract (method.md section 3)."""

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig
from ebt.core.energy import EnergyModel
from support import TanhEnergy, float64_loop


@pytest.mark.parametrize("sequence", [False, True], ids=["per-example", "per-position"])
def test_batched_step_equals_steps_on_single_examples(sequence):
    """Summing energies over the batch gives each example its own gradient (method.md 3.1)."""
    leading = (5, 3) if sequence else (5,)
    model = TanhEnergy(context_dim=2, candidate_dim=4, hidden=6, seed=13).double()
    generator = torch.Generator().manual_seed(14)
    context = torch.randn(*leading, 2, dtype=torch.float64, generator=generator)
    initial = torch.randn(*leading, 4, dtype=torch.float64, generator=generator)
    loop = float64_loop(ThinkingConfig(step_size=0.4))
    config = InferenceConfig(steps=2)

    batched = loop.infer(model, context, initial, config).candidates[-1]
    one_by_one = torch.cat(
        [
            loop.infer(model, context[i : i + 1], initial[i : i + 1], config).candidates[-1]
            for i in range(5)
        ]
    )
    torch.testing.assert_close(batched, one_by_one, rtol=1e-14, atol=1e-14)


class WrongShapeEnergy(EnergyModel):
    candidate_event_dims = 1

    def energy(self, context, encoded):
        return encoded.sum()


def test_energy_of_the_wrong_shape_is_rejected():
    loop = float64_loop(ThinkingConfig(step_size=0.1))
    with pytest.raises(ValueError, match="energy has shape"):
        loop.infer(WrongShapeEnergy(), None, torch.ones(3, 2), InferenceConfig(steps=1))
