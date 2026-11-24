"""Two-branch regression data (method.md section 16.1)."""

import torch

from ebt.config import RegressionDataConfig
from ebt.tasks.regression import make_data

CONFIG = RegressionDataConfig(train_size=4000, validation_size=500, test_size=600, ood_size=300)


def test_split_sizes():
    data = make_data(CONFIG, seed=0)
    sizes = (len(data.train), len(data.validation), len(data.test), len(data.ood))
    assert sizes == (4000, 500, 600, 300)


def test_x_ranges():
    data = make_data(CONFIG, seed=0)
    for split in (data.train, data.validation, data.test):
        assert split.context.min() >= 0.25 and split.context.max() <= 4.0
    assert data.ood.context.min() >= 4.0 and data.ood.context.max() <= 9.0


def test_targets_lie_on_the_branches_with_balanced_signs():
    """Balance: the positive fraction is within 3 binomial standard deviations of 1/2."""
    data = make_data(CONFIG, seed=0)
    train = data.train
    torch.testing.assert_close(train.target.abs(), train.context.sqrt())
    positive = (train.target > 0).float().mean().item()
    assert abs(positive - 0.5) <= 3 * (0.25 / len(train)) ** 0.5
    assert abs(data.init_std - train.target.std().item()) < 1e-7


def test_same_seed_gives_identical_data_and_different_seed_does_not():
    first, second, other = make_data(CONFIG, 0), make_data(CONFIG, 0), make_data(CONFIG, 1)
    assert torch.equal(first.train.context, second.train.context)
    assert torch.equal(first.test.target, second.test.target)
    assert not torch.equal(first.train.context, other.train.context)


def test_splits_are_drawn_independently():
    data = make_data(CONFIG, seed=0)
    assert not torch.equal(data.validation.context, data.test.context[:500])
