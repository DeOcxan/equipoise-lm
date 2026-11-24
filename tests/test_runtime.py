"""Device selection without fallback, and seeding (docs/numerics.md)."""

import pytest
import torch

from ebt.runtime import make_generator, seed_everything, select_device


def test_cpu_is_selected():
    assert select_device("cpu") == torch.device("cpu")


def test_unavailable_mps_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        select_device("mps")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        select_device("cuda")


def test_seeding_makes_draws_repeat():
    seed_everything(5)
    first = torch.rand(3)
    seed_everything(5)
    assert torch.equal(first, torch.rand(3))
    a = torch.rand(3, generator=make_generator(7, torch.device("cpu")))
    b = torch.rand(3, generator=make_generator(7, torch.device("cpu")))
    assert torch.equal(a, b)
