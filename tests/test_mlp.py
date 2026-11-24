"""MLP energy and regressor (method.md section 16.2)."""

import torch

from ebt.config import InferenceConfig, MLPConfig, ThinkingConfig
from ebt.models.mlp import MLPEnergy, MLPRegressor, parameter_count
from support import float64_loop

CONFIG = MLPConfig(widths=(64, 64))
REGRESSOR = MLPConfig(widths=(65, 64))


def test_shapes():
    energy = MLPEnergy(1, 1, CONFIG)
    regressor = MLPRegressor(1, 1, CONFIG)
    x, y = torch.zeros(7, 1), torch.zeros(7, 1)
    assert energy.energy(x, y).shape == (7,)
    assert regressor(x).shape == (7, 1)


def test_parameter_counts_match_within_one_percent():
    energy, regressor = (
        parameter_count(MLPEnergy(1, 1, CONFIG)),
        parameter_count(MLPRegressor(1, 1, REGRESSOR)),
    )
    assert (energy, regressor) == (4416, 4419)
    assert abs(energy - regressor) / regressor < 0.01


def test_batched_step_equals_steps_on_single_examples():
    """The energy contract: no example's gradient depends on another example (method.md 3)."""
    model = MLPEnergy(1, 1, CONFIG).double()
    generator = torch.Generator().manual_seed(30)
    context = torch.rand(6, 1, dtype=torch.float64, generator=generator) * 4
    initial = torch.randn(6, 1, dtype=torch.float64, generator=generator)
    loop = float64_loop(ThinkingConfig(step_size=0.3))
    config = InferenceConfig(steps=3)
    batched = loop.infer(model, context, initial, config).candidates[-1]
    single = torch.cat(
        [
            loop.infer(model, context[i : i + 1], initial[i : i + 1], config).candidates[-1]
            for i in range(6)
        ]
    )
    torch.testing.assert_close(batched, single, rtol=1e-14, atol=1e-14)
