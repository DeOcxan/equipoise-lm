"""The contract between a task model and the thinking loop (method.md section 3)."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class EnergyModel(nn.Module, ABC):
    """Maps a context and a candidate to one energy per example or per position (method.md 3).

    The candidate has shape (*energy_shape, *event_shape), where event_shape is the trailing
    `candidate_event_dims` dimensions covered by one energy value. Examples: regression (B, D)
    and text logits (B, S, V) use 1; images (B, C, H, W) use 3.

    The loop takes the gradient of the sum of all energies in a batch (method.md section 3.1).
    That gradient equals the per-example gradients only if the energy of each example is
    independent of every other example. Subclasses must therefore not mix information across
    the energy elements of a batch: no batch normalization, no attention or pooling across
    examples, and, for sequences, no dependence of one position's energy on another position's
    candidate.
    """

    candidate_event_dims: int

    @abstractmethod
    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        """Energies of the encoded candidates, shape `energy_shape(candidate)` (method.md 3)."""

    def transform_candidate(self, candidate: Tensor) -> Tensor:
        """The thinking variable of a (noisy) candidate (method.md section 4.1).

        The thinking loop takes the energy gradient with respect to this variable and steps
        from it: ŷ ← after_step(v − α ∇_v E) with v = transform_candidate(ŷ + η). Identity by
        default; the text EBT uses softmax over the vocabulary.
        """
        return candidate

    def encode_candidate(self, variable: Tensor) -> Tensor:
        """Map from the thinking variable to the input of `energy` (method.md section 4).

        Identity by default. The thinking gradient is taken with respect to the variable, not
        its encoding.
        """
        return variable

    def after_step(self, candidate: Tensor) -> Tensor:
        """Transform applied after every update, including the last (method.md section 4.1).

        Identity by default.
        """
        return candidate

    def energy_shape(self, candidate: Tensor) -> torch.Size:
        """Shape of the energy: one value per example or per position (method.md 3.1)."""
        return candidate.shape[: candidate.dim() - self.candidate_event_dims]
