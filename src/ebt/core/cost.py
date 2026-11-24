"""Pass counting for cost comparisons (method.md section 6.3)."""

from dataclasses import dataclass


@dataclass
class PassCount:
    """Passes per input example, counted where the passes are made.

    One batched pass counts as one pass for every example in the batch. Best-of-M selection
    evaluates M candidates per input and therefore costs M times the passes of one candidate.
    """

    forward: int = 0
    """Energy evaluations."""
    backward: int = 0
    """First-order backward passes: gradients of the energy with respect to the candidate, or,
    for a feed-forward model, the backward pass of its loss."""
    second_order: int = 0
    """Steps whose candidate gradient keeps its graph. Each costs one Hessian-vector product,
    plus the backward pass through that step's energy, when the training loss is backpropagated."""

    def __add__(self, other: "PassCount") -> "PassCount":
        """Counts of two sets of passes together (method.md section 6.3)."""
        return PassCount(
            self.forward + other.forward,
            self.backward + other.backward,
            self.second_order + other.second_order,
        )

    def __mul__(self, factor: int) -> "PassCount":
        """Counts for `factor` candidates per input, as in best-of-M (method.md section 6.3)."""
        return PassCount(self.forward * factor, self.backward * factor, self.second_order * factor)
