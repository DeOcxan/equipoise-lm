"""Derivative-free optimizer of Florence et al. (2021), Algorithm 1 (method.md section 16.8)."""

import torch
from torch import Tensor

from ebt.core.energy import EnergyModel
from ebt.inference.dfo import DFOConfig, derivative_free_optimization


class Quadratic(EnergyModel):
    """E(x, y) = 20 (y - x)**2, minimum at y = x."""

    candidate_event_dims = 1

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        return 20 * ((encoded - context) ** 2).sum(-1)


def test_finds_the_minimum_of_a_quadratic():
    context = torch.tensor([[-0.7], [0.0], [1.2]])
    chosen, cost = derivative_free_optimization(
        Quadratic(), context, (-2.0, 2.0), DFOConfig(samples=2048), torch.Generator().manual_seed(0)
    )
    torch.testing.assert_close(chosen, context, atol=0.02, rtol=0)
    assert cost.forward == 3 * 2048 * 3 and cost.backward == 0


def test_candidates_stay_within_the_bounds():
    """A minimum outside the bounds is approached from inside and never passed."""
    context = torch.tensor([[5.0], [-5.0]])
    chosen, _ = derivative_free_optimization(
        Quadratic(), context, (-1.0, 1.0), DFOConfig(samples=512), torch.Generator().manual_seed(1)
    )
    assert chosen.abs().max().item() <= 1.0
    torch.testing.assert_close(chosen, torch.tensor([[1.0], [-1.0]]), atol=0.02, rtol=0)


def test_same_seed_gives_the_same_result():
    context = torch.randn(4, 1, generator=torch.Generator().manual_seed(2))
    results = [
        derivative_free_optimization(
            Quadratic(),
            context,
            (-2.0, 2.0),
            DFOConfig(samples=256),
            torch.Generator().manual_seed(3),
        )[0]
        for _ in range(2)
    ]
    assert torch.equal(*results)
