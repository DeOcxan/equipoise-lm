"""Energies with known minima, used to check the thinking loop (questions.md, G1)."""

import torch
from torch import Tensor, nn

from ebt.core.energy import EnergyModel


class QuadraticEnergy(EnergyModel):
    """E(y) = ½ (y − c)ᵀ A (y − c) over the last dimension, with A symmetric positive definite.

    The minimizer is c. Gradient descent with step size α converges for α < 2 / L, where L is
    the largest eigenvalue of A. A and c are parameters; the context is ignored.
    """

    candidate_event_dims = 1

    def __init__(self, precision: Tensor, center: Tensor) -> None:
        super().__init__()
        if precision.dim() != 2 or precision.shape[0] != precision.shape[1]:
            raise ValueError("precision must be a square matrix")
        if not torch.allclose(precision, precision.T):
            raise ValueError("precision must be symmetric")
        if center.shape != precision.shape[:1]:
            raise ValueError("center must have one entry per row of precision")
        self.precision = nn.Parameter(precision.clone())
        self.center = nn.Parameter(center.clone())

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        offset = encoded - self.center
        return 0.5 * torch.einsum("...i,ij,...j->...", offset, self.precision, offset)

    def minimizer(self) -> Tensor:
        """The minimizer c, used by the convergence tests (questions.md, G1)."""
        return self.center.detach()

    def lipschitz(self) -> float:
        """Largest eigenvalue of A, the Lipschitz constant of the gradient (questions.md, G1)."""
        return torch.linalg.eigvalsh(self.precision.detach()).max().item()


class DoubleWellEnergy(EnergyModel):
    """E(y) = Σ_j (y_j² − 1)² over the last dimension.

    Each coordinate has minima at −1 and +1 separated by a barrier at 0. The context is ignored.
    """

    candidate_event_dims = 1

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        return ((encoded**2 - 1) ** 2).sum(dim=-1)

    @staticmethod
    def basin_minimum(start: Tensor) -> Tensor:
        """The minimum of the basin containing `start`: the sign of each coordinate (G1)."""
        if (start == 0).any():
            raise ValueError("a start point on the barrier belongs to no basin")
        return torch.sign(start)
