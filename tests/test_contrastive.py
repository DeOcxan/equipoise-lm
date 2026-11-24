"""Contrastive energy loss on hand-computed values (method.md section 16.4)."""

import math

import pytest
import torch
import torch.nn.functional as F

from ebt.core.contrastive import contrastive_loss, energy_magnitude, info_nce


def test_single_pair():
    """E_real = 1, E_fake = 3: softplus(1 - 3) = log(1 + e**-2) = 0.126928011..."""
    loss = contrastive_loss(
        torch.tensor([1.0], dtype=torch.float64), torch.tensor([3.0], dtype=torch.float64)
    )
    assert loss.item() == pytest.approx(math.log1p(math.exp(-2.0)), abs=1e-15)
    assert loss.item() == pytest.approx(0.126928011, abs=1e-9)


def test_batch_mean_equals_two_class_cross_entropy():
    """The mean over pairs equals the cross-entropy of the logits (-E_real, -E_fake), class 0."""
    real = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
    fake = torch.tensor([3.0, 0.0, -1.0], dtype=torch.float64)
    expected = (math.log1p(math.exp(-2.0)) + math.log(2.0) + math.log1p(math.exp(-1.0))) / 3
    loss = contrastive_loss(real, fake)
    assert loss.item() == pytest.approx(expected, abs=1e-15)
    logits = -torch.stack([real, fake], dim=1)
    target = torch.zeros(3, dtype=torch.long)
    assert loss.item() == pytest.approx(F.cross_entropy(logits, target).item(), abs=1e-15)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        contrastive_loss(torch.zeros(2), torch.zeros(3))


def test_energy_magnitude_on_hand_values():
    """mean(E_real**2 + E_fake**2) for (1, 3) and (-2, 0): ((1 + 9) + (4 + 0)) / 2 = 7."""
    real = torch.tensor([1.0, -2.0], dtype=torch.float64)
    fake = torch.tensor([3.0, 0.0], dtype=torch.float64)
    assert energy_magnitude(real, fake).item() == pytest.approx(7.0, abs=1e-15)
    with pytest.raises(ValueError):
        energy_magnitude(torch.zeros(2), torch.zeros(3))


def test_info_nce_on_hand_values():
    """E_real = 0, counter energies 1 and 2:
    -log(1 / (1 + e**-1 + e**-2)) = log(1 + e**-1 + e**-2)."""
    real = torch.tensor([0.0], dtype=torch.float64)
    counter = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    expected = math.log(1 + math.exp(-1.0) + math.exp(-2.0))
    assert info_nce(real, counter).item() == pytest.approx(expected, abs=1e-15)
    assert info_nce(real, counter).item() == pytest.approx(0.4076059644, abs=1e-9)


def test_info_nce_with_one_counter_example_equals_the_contrastive_loss():
    generator = torch.Generator().manual_seed(3)
    real = torch.randn(5, dtype=torch.float64, generator=generator)
    fake = torch.randn(5, dtype=torch.float64, generator=generator)
    torch.testing.assert_close(info_nce(real, fake.unsqueeze(-1)), contrastive_loss(real, fake))


def test_info_nce_equals_cross_entropy_with_the_true_sample_last():
    """The authors' code puts the true sample in the last column; the position does not matter."""
    generator = torch.Generator().manual_seed(4)
    real = torch.randn(4, dtype=torch.float64, generator=generator)
    counter = torch.randn(4, 6, dtype=torch.float64, generator=generator)
    logits = -torch.cat([counter, real.unsqueeze(-1)], dim=-1)
    expected = F.cross_entropy(logits, torch.full((4,), 6, dtype=torch.long))
    torch.testing.assert_close(info_nce(real, counter), expected)
    with pytest.raises(ValueError):
        info_nce(real, counter[:3])
