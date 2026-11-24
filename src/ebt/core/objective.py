"""Training loss over a thinking trajectory (method.md sections 6.1 and 6.2)."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from ebt.config import TrainingMode


def trajectory_loss(
    predictions: Sequence[Tensor],
    target: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    mode: TrainingMode,
) -> Tensor:
    """Loss of the predictions ŷ_1, ..., ŷ_k against the target (method.md section 6.2).

    S1 averages `loss_fn` over all k predictions; S2 and full unroll use the final prediction
    only. `loss_fn` returns a scalar, for example cross-entropy for text or Smooth L1 for
    images (method.md section 6.1).
    """
    if not predictions:
        raise ValueError("predictions must not be empty")
    if mode is TrainingMode.S1:
        return torch.stack([loss_fn(prediction, target) for prediction in predictions]).mean()
    return loss_fn(predictions[-1], target)
