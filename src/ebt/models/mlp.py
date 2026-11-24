"""MLP energy and MLP regressor of matching size (method.md section 16)."""

import torch
from torch import Tensor, nn

from ebt.config import MLPConfig
from ebt.core.energy import EnergyModel

ACTIVATION_MODULES = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(input_dim: int, output_dim: int, config: MLPConfig, output_bias: bool) -> nn.Sequential:
    """Hidden layers of widths `config.widths` with `config.activation`, then a linear output.

    SiLU rather than ReLU for the EBTs: with ReLU the energy is piecewise linear in the
    candidate, so its second derivative with respect to the candidate is zero almost everywhere.
    The mixed derivative ∂(∇_ŷ E)/∂θ that S2 trains through does not vanish, but the curvature
    in ŷ does, and it is needed when the loss is backpropagated through several steps (S1, full
    unroll). SiLU gives an energy that is smooth in ŷ.
    """
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in config.widths:
        layers += [nn.Linear(width, hidden), ACTIVATION_MODULES[config.activation]()]
        width = hidden
    layers.append(nn.Linear(width, output_dim, bias=output_bias))
    return nn.Sequential(*layers)


class MLPEnergy(EnergyModel):
    """E(x, ŷ) = MLP([x; ŷ]), one scalar per example (method.md sections 3 and 16).

    The output layer has no bias, because a constant offset does not change the minimum
    (method.md section 3.1). Every example is processed independently, as the energy contract
    requires.
    """

    candidate_event_dims = 1

    def __init__(self, context_dim: int, candidate_dim: int, config: MLPConfig) -> None:
        super().__init__()
        self.net = mlp(context_dim + candidate_dim, 1, config, output_bias=False)

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        """Energy of each example, shape (B,) (method.md section 3)."""
        return self.net(torch.cat([context, encoded], dim=-1)).squeeze(-1)


class MLPRegressor(nn.Module):
    """ŷ = MLP(x), the feed-forward baseline of the same size (method.md section 16)."""

    def __init__(self, input_dim: int, output_dim: int, config: MLPConfig) -> None:
        super().__init__()
        self.net = mlp(input_dim, output_dim, config, output_bias=True)

    def forward(self, context: Tensor) -> Tensor:
        return self.net(context)


def parameter_count(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
