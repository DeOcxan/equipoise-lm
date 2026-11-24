"""Batched prediction for evaluation (method.md sections 5, 16 and 17)."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ebt.config import InferenceConfig
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.core.thinking import ThinkingLoop
from ebt.training.batch import Batch


@dataclass
class Predictions:
    """Predictions for every row of an evaluation set, on the CPU."""

    initial: Tensor | None
    """Initial candidates ŷ_0 of an EBT; None for a feed-forward model."""
    final: Tensor
    """Final predictions."""
    energy: Tensor | None
    """E(x, ŷ_k) of the final candidates; None for a feed-forward model."""
    last_step_energy: Tensor | None
    """Energy at the input of the final step, E(x, ŷ_{k−1}); None for a feed-forward model."""
    cost: PassCount
    """Total example-passes."""


def predict_ebt(
    model: EnergyModel,
    loop: ThinkingLoop,
    data: Batch,
    initial: Tensor,
    config: InferenceConfig,
    generator: torch.Generator | None,
    batch_size: int,
    final_energy: bool = True,
) -> Predictions:
    """Think from `initial` for every row of `data` (method.md section 5.1).

    Uses `ThinkingLoop.infer`, so there is no step-size or step-count randomness and the noise
    is `config.langevin_std`. With `final_energy` False the energy of the final candidates is
    not evaluated, which saves one forward pass per row, and `Predictions.energy` is None.
    """
    finals, energies, last_step, cost = [], [], [], PassCount()
    for start in range(0, len(data), batch_size):
        rows = slice(start, start + batch_size)
        chunk = data.rows(rows)
        trajectory = loop.infer(
            model, chunk.context, initial[rows], config, generator, final_energy=final_energy
        )
        finals.append(trajectory.candidates[-1].cpu())
        if final_energy:
            energies.append(trajectory.final_energy.cpu())
        last_step.append(trajectory.energies[-1].cpu())
        cost = cost + trajectory.cost * len(chunk)
    return Predictions(
        initial=initial.detach().cpu(),
        final=torch.cat(finals),
        energy=torch.cat(energies) if final_energy else None,
        last_step_energy=torch.cat(last_step),
        cost=cost,
    )


def predict_feedforward(model: nn.Module, inputs: Tensor, batch_size: int) -> Predictions:
    """Forward pass of a feed-forward model for every row of `inputs`: one pass per row."""
    with torch.no_grad():
        finals = [
            model(inputs[start : start + batch_size]).cpu()
            for start in range(0, len(inputs), batch_size)
        ]
    return Predictions(
        initial=None,
        final=torch.cat(finals),
        energy=None,
        last_step_energy=None,
        cost=PassCount(forward=len(inputs)),
    )
