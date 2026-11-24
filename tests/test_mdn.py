"""Mixture density network on hand-computed values (method.md section 16.8, Bishop 1994)."""

import math

import pytest
import torch

from ebt.config import MLPConfig
from ebt.experiments.regression_g2 import MDN_WIDTHS
from ebt.models.mdn import (
    Mixture,
    MixtureDensityNetwork,
    log_density,
    most_probable_branch,
    most_probable_value,
    negative_log_likelihood,
    sampled_centre,
)
from ebt.models.mlp import parameter_count


def mixture(weights, means, scales) -> Mixture:
    def row(values):
        return torch.tensor([values], dtype=torch.float64)

    return Mixture(row(weights).log(), row(means), row(scales).log())


def gaussian(y, mean, scale):
    return math.exp(-0.5 * ((y - mean) / scale) ** 2) / (scale * math.sqrt(2 * math.pi))


def test_negative_log_likelihood_of_two_kernels():
    """α = (0.25, 0.75), μ = (-1, 2), σ = (1, 0.5), y = 0.5 (Bishop's eq. 22, 23, 29)."""
    m = mixture([0.25, 0.75], [-1.0, 2.0], [1.0, 0.5])
    target = torch.tensor([[0.5]], dtype=torch.float64)
    expected = -math.log(0.25 * gaussian(0.5, -1, 1) + 0.75 * gaussian(0.5, 2, 0.5))
    assert negative_log_likelihood(m, target).item() == pytest.approx(expected, abs=1e-14)
    assert log_density(m, target).item() == pytest.approx(-expected, abs=1e-14)


def test_readouts_pick_the_kernel_of_eq_49_and_eq_48():
    """α = (0.6, 0.4), σ = (1, 0.25): eq. 49 picks kernel 0 (largest α); eq. 48 picks
    kernel 1 (largest α/σ: 0.6 against 1.6)."""
    m = mixture([0.6, 0.4], [-1.5, 1.5], [1.0, 0.25])
    assert most_probable_branch(m).item() == -1.5
    assert most_probable_value(m).item() == 1.5


def test_sampled_centre_follows_the_mixing_coefficients():
    m = mixture([0.3, 0.7], [-1.0, 1.0], [1.0, 1.0])
    rows = Mixture(*(tensor.expand(20_000, -1) for tensor in vars(m).values()))
    draws = sampled_centre(rows, torch.Generator().manual_seed(5))
    assert (draws > 0).double().mean().item() == pytest.approx(0.7, abs=0.02)
    again = sampled_centre(rows, torch.Generator().manual_seed(5))
    assert torch.equal(draws, again)


def test_network_outputs_a_normalized_mixture():
    model = MixtureDensityNetwork(1, 5, MLPConfig(widths=(8, 8), activation="tanh"))
    out = model(torch.randn(4, 1))
    assert out.means.shape == out.log_weights.shape == out.log_scales.shape == (4, 5)
    torch.testing.assert_close(out.log_weights.exp().sum(-1), torch.ones(4))


@pytest.mark.parametrize(("components", "expected"), [(2, 4416), (5, 4420), (10, 4414)])
def test_parameter_counts_within_one_percent_of_the_energy(components, expected):
    config = MLPConfig(widths=MDN_WIDTHS[components], activation="tanh")
    count = parameter_count(MixtureDensityNetwork(1, components, config))
    assert count == expected
    assert abs(count - 4416) / 4416 < 0.01
