"""Replay buffer of final candidates (method.md section 9)."""

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from ebt.config import ReplayConfig


@dataclass
class ReplayBatch:
    """A fresh batch followed by replayed rows. Rows 0..fresh-1 are fresh."""

    context: Tensor
    target: Tensor
    initial: Tensor
    """Initial candidates: the fresh initialization for fresh rows, the stored final candidate
    for replayed rows."""
    fresh: int


class ReplayBuffer:
    """First-in-first-out store of (context, target, final candidate) rows (method.md section 9).

    Rows are single examples along dimension 0 of any tensor shape. They are stored detached
    and cloned.
    """

    def __init__(self, config: ReplayConfig, generator: torch.Generator | None = None) -> None:
        self.config = config
        self._generator = generator
        self._rows: deque[tuple[Tensor, Tensor, Tensor]] = deque(maxlen=config.capacity)

    def __len__(self) -> int:
        return len(self._rows)

    def extend(self, context: Tensor, target: Tensor, initial: Tensor) -> ReplayBatch:
        """Append min(int(fraction · B), len(buffer)) stored rows to B fresh rows (method.md 9).

        The rows are drawn without replacement. Each replayed row starts from its stored final
        candidate, so its optimization continues where an earlier training step left it.
        """
        fresh = initial.shape[0]
        count = min(int(self.config.fraction * fresh), len(self._rows))
        if count == 0:
            return ReplayBatch(context, target, initial, fresh)
        order = torch.randperm(len(self._rows), generator=self._generator)[:count].tolist()
        rows = [self._rows[index] for index in order]
        return ReplayBatch(
            context=torch.cat([context, torch.stack([row[0] for row in rows])]),
            target=torch.cat([target, torch.stack([row[1] for row in rows])]),
            initial=torch.cat([initial, torch.stack([row[2] for row in rows])]),
            fresh=fresh,
        )

    def store(self, batch: ReplayBatch, final: Tensor) -> None:
        """Store the fresh rows of `batch` with their final candidates (method.md section 9).

        Replayed rows are not stored again. Beyond capacity the oldest rows are dropped.
        """
        for index in range(batch.fresh):
            self._rows.append(
                (
                    batch.context[index].detach().clone(),
                    batch.target[index].detach().clone(),
                    final[index].detach().clone(),
                )
            )
