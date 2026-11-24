"""The training loop shared by every objective and task (method.md sections 6 and 10)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ebt.config import TrainingConfig
from ebt.core.cost import PassCount
from ebt.training.batch import Batch, shuffled_batches
from ebt.training.objectives import TrainingObjective
from ebt.training.optimizer import build_optimizer, build_schedule

HISTORY_POINTS = 100
"""Number of points of the recorded training-loss curve."""


@dataclass
class TrainingHistory:
    """Mean training loss over consecutive intervals, and the total cost of training."""

    steps: list[int] = field(default_factory=list)
    """Last step of each interval."""
    train_loss: list[float] = field(default_factory=list)
    cost: PassCount = field(default_factory=PassCount)
    """Total example-passes of all training steps."""
    examples: int = 0
    """Fresh examples processed."""
    validation_steps: list[int] = field(default_factory=list)
    """Steps after which the validation metrics were measured."""
    validation: dict[str, list[float]] = field(default_factory=dict)
    """Each validation metric at `validation_steps`."""


def train(
    objective: TrainingObjective,
    data: Batch,
    config: TrainingConfig,
    batch_generator: torch.Generator,
    noise_generator: torch.Generator,
    validate: Callable[[], dict[str, float]] | None = None,
    validate_every: int = 0,
) -> TrainingHistory:
    """Train `objective.module` on `data` for `config.train_steps` steps.

    Each step draws a batch with `batch_generator` (CPU), computes the objective's loss with
    `noise_generator` (on the device of the data), clips the gradient norm of all parameters to
    `config.gradient_clip`, and advances AdamW and the warm-up and cosine schedule.

    With `validate`, it is called after every `validate_every` steps, in evaluation mode, and
    its metrics are recorded as the validation curve. It must not draw from `noise_generator`.
    """
    objective.module.train()
    optimizer = build_optimizer(objective.module, objective.loop, config)
    schedule = build_schedule(optimizer, config)
    parameters = [p for p in objective.module.parameters() if p.requires_grad]
    history = TrainingHistory()
    interval = max(1, config.train_steps // HISTORY_POINTS)
    batches = shuffled_batches(data, config.batch_size, batch_generator)
    interval_loss = 0.0

    for step in range(1, config.train_steps + 1):
        batch = next(batches)
        result = objective.batch_loss(batch, noise_generator)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        schedule.step()

        history.cost = history.cost + result.cost
        history.examples += len(batch)
        interval_loss += result.loss.item()
        if step % interval == 0:
            history.steps.append(step)
            history.train_loss.append(interval_loss / interval)
            interval_loss = 0.0
        if validate is not None and step % validate_every == 0:
            objective.module.eval()
            history.validation_steps.append(step)
            for name, value in validate().items():
                history.validation.setdefault(name, []).append(value)
            objective.module.train()

    objective.module.eval()
    return history
