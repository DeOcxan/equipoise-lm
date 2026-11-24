"""Training loss of each mode on a hand-computed example (method.md section 6.2)."""

import pytest
import torch

from ebt.config import TrainingMode
from ebt.core.objective import trajectory_loss


def squared_error(prediction, target):
    return ((prediction - target) ** 2).sum()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # Predictions 1, 2, 4 against target 0 give per-step losses 1, 4, 16.
        (TrainingMode.S1, (1 + 4 + 16) / 3),
        (TrainingMode.S2, 16.0),
        (TrainingMode.FULL, 16.0),
    ],
)
def test_loss_of_each_mode(mode, expected):
    predictions = [torch.tensor([value], dtype=torch.float64) for value in (1.0, 2.0, 4.0)]
    loss = trajectory_loss(predictions, torch.zeros(1, dtype=torch.float64), squared_error, mode)
    assert loss.item() == pytest.approx(expected, abs=1e-15)


def test_empty_trajectory_is_rejected():
    with pytest.raises(ValueError):
        trajectory_loss([], torch.zeros(1), squared_error, TrainingMode.S1)
