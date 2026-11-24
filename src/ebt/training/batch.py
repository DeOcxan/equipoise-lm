"""A batch of (context, target) rows and its iteration, shared by every task."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Batch:
    """Rows along dimension 0: the context x and the target y of each example.

    The context is None for an unconditional energy, such as the image denoising energy
    (method.md section 3.2).
    """

    context: Tensor | None
    target: Tensor

    def __post_init__(self) -> None:
        if self.context is not None and self.context.shape[0] != self.target.shape[0]:
            raise ValueError("context and target must have the same number of rows")

    def __len__(self) -> int:
        return self.target.shape[0]

    def rows(self, index: Tensor | slice) -> "Batch":
        """The rows selected by `index`."""
        context = None if self.context is None else self.context[index]
        return Batch(context, self.target[index])

    def to(self, device: torch.device) -> "Batch":
        context = None if self.context is None else self.context.to(device)
        return Batch(context, self.target.to(device))

    def chunks(self, size: int) -> Iterator["Batch"]:
        """Consecutive chunks of at most `size` rows, in order."""
        for start in range(0, len(self), size):
            yield self.rows(slice(start, start + size))


def shuffled_batches(data: Batch, batch_size: int, generator: torch.Generator) -> Iterator[Batch]:
    """An endless stream of batches: each pass over the data uses a new random order.

    The permutation is drawn on the CPU with `generator`; rows that do not fill a whole batch
    at the end of a pass are dropped, so every batch has `batch_size` rows.
    """
    if batch_size > len(data):
        raise ValueError("batch_size exceeds the number of rows")
    while True:
        order = torch.randperm(len(data), generator=generator).to(data.target.device)
        for start in range(0, len(data) - batch_size + 1, batch_size):
            yield data.rows(order[start : start + batch_size])
