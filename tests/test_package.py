"""Checks that the package imports and that the installed torch supports second-order gradients."""

import re

import torch

import ebt


def test_version_is_semantic():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ebt.__version__)


def test_second_order_gradient_on_cpu():
    # E(y) = w * (y - 3)^2, one step y1 = y0 - a * dE/dy, loss L = y1^2.
    # With y0 = 0, w = 0.5, a = 0.1: dE/dy = 2w(y0 - 3) = -3, y1 = 0.3, L = 0.09.
    # dL/dw = 2 * y1 * dy1/dw = 2 * 0.3 * (-a * 2 * (y0 - 3)) = 2 * 0.3 * 0.6 = 0.36.
    y0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    energy = w * (y0 - 3.0) ** 2
    (grad_y,) = torch.autograd.grad(energy, y0, create_graph=True)
    y1 = y0 - 0.1 * grad_y
    loss = y1**2
    loss.backward()
    assert torch.isclose(y1.detach(), torch.tensor(0.3, dtype=torch.float64))
    assert torch.isclose(w.grad, torch.tensor(0.36, dtype=torch.float64))
