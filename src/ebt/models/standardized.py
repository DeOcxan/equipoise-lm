"""An energy trained on standardized data, evaluated in the original units (method.md 16.8)."""

import torch
from torch import Tensor

from ebt.core.energy import EnergyModel
from ebt.tasks.regression import Standardization


class StandardizedEnergy(EnergyModel):
    """E(x, y) = inner((x − μ_x) / σ_x, (y − μ_y) / σ_y) for scalar x and y.

    Only the first context column is standardized; further columns, such as a landscape
    index, pass unchanged. Gradients with respect to y are in the original units.
    """

    candidate_event_dims = 1

    def __init__(self, inner: EnergyModel, standardization: Standardization) -> None:
        super().__init__()
        self.inner = inner
        self.standardization = standardization

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        s = self.standardization
        x = (context[..., :1] - s.x_mean) / s.x_std
        context = torch.cat([x, context[..., 1:]], dim=-1)
        return self.inner.energy(context, (encoded - s.y_mean) / s.y_std)
