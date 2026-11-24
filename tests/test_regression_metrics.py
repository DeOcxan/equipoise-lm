"""G2 metrics on hand-computed predictions (method.md section 16.7)."""

import math

import pytest
import torch

from ebt.tasks.regression import branch_distance, regression_metrics
from ebt.training.batch import Batch


def test_branch_distance_on_a_branch_at_the_mean_and_halfway():
    x = torch.tensor([[4.0], [4.0], [4.0], [1.0]])
    prediction = torch.tensor([[2.0], [-2.0], [0.0], [0.5]])
    torch.testing.assert_close(
        branch_distance(x, prediction), torch.tensor([[0.0], [0.0], [1.0], [0.5]])
    )


def test_metrics_of_a_small_split():
    x = torch.tensor([[4.0], [4.0], [1.0], [1.0]])
    target = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]])
    prediction = torch.tensor([[2.1], [0.0], [-1.0], [0.5]])
    initial = torch.tensor([[1.0], [1.0], [-0.5], [-2.0]])
    metrics = regression_metrics(Batch(x, target), prediction, initial)

    # r = 0.05, 1, 0, 0.5
    assert metrics["branch_distance"] == pytest.approx((0.05 + 1 + 0 + 0.5) / 4)
    assert metrics["on_branch"] == pytest.approx(0.5)
    assert metrics["positive_branch"] == pytest.approx(0.5)
    assert metrics["positive_branch_on_branch"] == pytest.approx(0.5)  # 2.1 and -1.0
    assert metrics["mse"] == pytest.approx((0.01 + 4 + 4 + 2.25) / 4)
    assert metrics["mean_predictor_mse"] == pytest.approx((4 + 4 + 1 + 1) / 4)
    # signs: (+,+) agree, (0,+) differ, (-,-) agree, (+,-) differ
    assert metrics["initial_sign_agreement"] == pytest.approx(0.5)
    assert metrics["mean_move"] == pytest.approx((1.1 + 1 + 0.5 + 2.5) / 4)


def test_regressor_metrics_have_no_initial_candidate_entries():
    x, target = torch.tensor([[1.0]]), torch.tensor([[1.0]])
    metrics = regression_metrics(Batch(x, target), torch.tensor([[3.0]]), None)
    assert "initial_sign_agreement" not in metrics and "mean_move" not in metrics
    assert math.isnan(metrics["positive_branch_on_branch"])
