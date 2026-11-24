"""Convergence of the thinking loop on energies with known minima (G1)."""

import torch

from ebt.config import InferenceConfig, ThinkingConfig
from ebt.core.analytic import DoubleWellEnergy, QuadraticEnergy
from support import float64_loop


def test_quadratic_converges_at_the_gradient_descent_rate():
    """With α < 2/L the error contracts by rho = max_j |1 - αλ_j| per step.

    Eigenvalues {0.5, 1, 2} in a rotated basis and α = 0.9 < 2/L = 1 give rho = max(0.55, 0.1,
    0.8) = 0.8. The bound ‖y_n - c‖ ≤ rho**n ‖y_0 - c‖ is checked at every step; after 200 steps
    rho**200 ≈ 4e-20, so the error is below 1e-8.
    """
    generator = torch.Generator().manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator))
    eigenvalues = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    precision = rotation @ torch.diag(eigenvalues) @ rotation.T
    precision = (precision + precision.T) / 2
    center = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    model = QuadraticEnergy(precision, center)
    alpha = 0.9
    assert alpha < 2 / model.lipschitz()
    rate = max(abs(1 - alpha * value) for value in eigenvalues.tolist())

    initial = torch.randn(8, 3, dtype=torch.float64, generator=generator) * 3
    loop = float64_loop(ThinkingConfig(step_size=alpha))
    trajectory = loop.infer(model, None, initial, InferenceConfig(steps=200))

    start_error = (initial - center).norm(dim=-1)
    for n, candidate in enumerate(trajectory.candidates):
        error = (candidate - center).norm(dim=-1)
        assert (error <= rate**n * start_error + 1e-12).all(), n
    assert (trajectory.candidates[-1] - model.minimizer()).abs().max() < 1e-8


def test_double_well_converges_to_the_minimum_of_the_starting_basin():
    """Each coordinate reaches the minimum (±1) of the basin it starts in, without crossing 0.

    E(y) = (y**2 - 1)**2, E'(y) = 4y(y**2 - 1), α = 0.05, starts with |y| in [0.2, 1.5] in both
    basins. Because E is even, it suffices to argue for y > 0.

    Bound on the gradient over the start region: on [0.2, 1], |E'| ≤ max 4y(1 - y**2) = 8/(3√3)
    ≈ 1.54 (at y = 1/√3); on [1, 1.5], |E'| ≤ E'(1.5) = 7.5. So one step moves y by at most
    α·7.5 = 0.375, less than the distance 1 from any y ≥ 1 to the barrier at 0; for y < 1 the
    gradient is negative and the step moves y away from the barrier.

    Monotone convergence: the update maps y to f(y) = y(1.2 - 0.2y**2), which is increasing on
    [0, √2]. For y in [0.2, 1), f(y) > y and f(y) ≤ f(1) = 1; for y in (1, 1.5], f(y) < y and
    f(y) ≥ min(f(1), f(1.5)) = min(1, 1.125) = 1. Iterates therefore move monotonically
    towards 1 and never leave [0.2, 1.5]. Near 1, |f'(1)| = 0.6, so 400 steps reach 1 within
    1e-8 from every start.
    """
    starts = torch.tensor([0.2, 0.5, 0.9, 1.2, 1.5], dtype=torch.float64)
    initial = torch.stack([starts, -starts], dim=1)  # both basins in every row
    model = DoubleWellEnergy()
    loop = float64_loop(ThinkingConfig(step_size=0.05))
    trajectory = loop.infer(model, None, initial, InferenceConfig(steps=400))

    basin = model.basin_minimum(initial)
    for candidate in trajectory.candidates:
        assert (torch.sign(candidate) == basin).all()
        assert (candidate.abs() >= 0.2 - 1e-12).all() and (candidate.abs() <= 1.5 + 1e-12).all()
    assert (trajectory.candidates[-1] - basin).abs().max() < 1e-8
