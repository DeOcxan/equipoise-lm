"""Loss of one training batch for an EBT, an InfoNCE energy and a feed-forward model.

Method: method.md sections 6, 16 and 17.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ebt.config import ObjectiveConfig
from ebt.core.contrastive import contrastive_loss, energy_magnitude, info_nce
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.core.objective import trajectory_loss
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop
from ebt.training.batch import Batch

LossFn = Callable[[Tensor, Tensor], Tensor]
InitialFn = Callable[[Batch, torch.Generator], Tensor]
InputFn = Callable[[Batch, torch.Generator], Tensor]


@dataclass
class BatchLoss:
    """Training loss of one batch and the passes it took, in example-passes.

    An example-pass is one pass for one row: a batched pass over n rows counts n.
    """

    loss: Tensor
    cost: PassCount


class TrainingObjective(ABC):
    """What the trainer optimizes: the trainable module and the loss of one batch."""

    module: nn.Module
    """Everything that is saved and optimized, including the thinking loop of an EBT."""
    loop: ThinkingLoop | None
    """The thinking loop, whose α the optimizer puts in its own group; None for a baseline."""

    @abstractmethod
    def batch_loss(self, batch: Batch, generator: torch.Generator) -> BatchLoss:
        """Loss of one batch; `generator` supplies initial candidates and thinking noise."""


class EBTObjective(TrainingObjective):
    """Thinking, then the weighted reconstruction and contrastive losses (method.md 6.2, 16).

    With a replay buffer, each fresh batch is extended with replayed rows that restart from
    their stored candidates, and the fresh rows are stored afterwards (method.md section 9).
    The losses are computed over all rows, fresh and replayed.
    """

    def __init__(
        self,
        model: EnergyModel,
        loop: ThinkingLoop,
        loss_fn: LossFn,
        objective: ObjectiveConfig,
        initial: InitialFn,
        replay: ReplayBuffer | None = None,
    ) -> None:
        self.model = model
        self.loop = loop
        self.module = nn.ModuleList([model, loop])
        self.loss_fn = loss_fn
        self.objective = objective
        self.initial = initial
        self.replay = replay

    def batch_loss(self, batch: Batch, generator: torch.Generator) -> BatchLoss:
        initial = self.initial(batch, generator)
        context, target = batch.context, batch.target
        replayed = None
        if self.replay is not None:
            replayed = self.replay.extend(context, target, initial)
            context, target, initial = replayed.context, replayed.target, replayed.initial
        rows = initial.shape[0]

        trajectory = self.loop.train_trajectory(self.model, context, initial, generator)
        cost = trajectory.cost * rows
        loss = torch.zeros((), dtype=initial.dtype, device=initial.device)
        if self.objective.reconstruction_weight > 0:
            reconstruction = trajectory_loss(
                trajectory.candidates[1:], target, self.loss_fn, self.loop.config.mode
            )
            loss = loss + self.objective.reconstruction_weight * reconstruction
        contrastive_weight = self.objective.contrastive_weight
        regularization_weight = self.objective.energy_regularization_weight
        if contrastive_weight > 0 or regularization_weight > 0:
            # Fake energy: the energy at the input of the final step, which keeps its graph
            # (method.md section 16.4). Real energy: one extra forward and backward pass.
            real = self.model.energy(context, self.model.encode_candidate(target))
            fake = trajectory.energies[-1]
            if contrastive_weight > 0:
                loss = loss + contrastive_weight * contrastive_loss(real, fake)
            if regularization_weight > 0:
                loss = loss + regularization_weight * energy_magnitude(real, fake)
            cost = cost + PassCount(forward=1, backward=1) * rows

        if replayed is not None:
            self.replay.store(replayed, trajectory.candidates[-1])
        return BatchLoss(loss, cost)


class InfoNCEObjective(TrainingObjective):
    """InfoNCE with uniform counter-examples (method.md section 16.8; Florence et al. 2021, §2).

    For every row, `counter_examples` candidates are drawn uniformly from `bounds` and the loss
    is the cross-entropy of the logits −E over the true target and those candidates, with the
    true target as the correct class. Each row costs one forward and one backward pass for each
    of its counter_examples + 1 energies.
    """

    def __init__(
        self, model: EnergyModel, counter_examples: int, bounds: tuple[float, float]
    ) -> None:
        self.model = model
        self.loop = None
        self.module = model
        self.counter_examples = counter_examples
        self.bounds = bounds

    def batch_loss(self, batch: Batch, generator: torch.Generator) -> BatchLoss:
        target, count = batch.target, self.counter_examples
        low, high = self.bounds
        uniform = torch.rand(
            (len(batch), count, *target.shape[1:]),
            generator=generator,
            dtype=target.dtype,
            device=target.device,
        )
        counter = low + (high - low) * uniform
        context = batch.context.unsqueeze(1).expand(-1, count, *batch.context.shape[1:])
        counter_energy = self.model.energy(context, self.model.encode_candidate(counter))
        real_energy = self.model.energy(batch.context, self.model.encode_candidate(target))
        passes = PassCount(forward=count + 1, backward=count + 1) * len(batch)
        return BatchLoss(info_nce(real_energy, counter_energy), passes)


class FeedForwardObjective(TrainingObjective):
    """A feed-forward model trained on the loss of its output: one forward, one backward pass.

    `inputs` builds the model input of a batch, for example a freshly noised image; by default
    the input is the context. `loss_fn` compares the output with the target; for a mixture
    density network the output is the mixture's parameters and the loss its negative
    log-likelihood.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable, inputs: InputFn | None = None) -> None:
        self.model = model
        self.loop = None
        self.module = model
        self.loss_fn = loss_fn
        self.inputs = inputs

    def batch_loss(self, batch: Batch, generator: torch.Generator) -> BatchLoss:
        inputs = batch.context if self.inputs is None else self.inputs(batch, generator)
        output = self.model(inputs)
        return BatchLoss(
            self.loss_fn(output, batch.target), PassCount(forward=1, backward=1) * len(batch)
        )
