"""Optimizer and learning-rate schedule shared by every task (method.md sections 10 and 16)."""

import math

import torch
from torch import nn

from ebt.config import TrainingConfig
from ebt.core.thinking import ThinkingLoop

BETAS = (0.9, 0.999)
"""AdamW momentum coefficients (Table D.2 of the paper)."""


def build_optimizer(
    model: nn.Module, loop: ThinkingLoop | None, config: TrainingConfig
) -> torch.optim.AdamW:
    """AdamW over the model, with a learnable α in its own group (method.md section 10).

    The α group has learning rate `learning_rate · step_size_lr_multiplier` and no weight
    decay; all other trainable parameters use `learning_rate` and `weight_decay`.
    """
    alpha = loop.step_size.raw if loop is not None else None
    groups = [
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and parameter is not alpha
            ],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]
    if alpha is not None and alpha.requires_grad:
        groups.append(
            {
                "params": [alpha],
                "lr": config.learning_rate * config.step_size_lr_multiplier,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups, betas=BETAS)


def build_schedule(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to 1 / min_lr_scale of each group's rate (Table D.2).

    The factor rises linearly from 1 / warmup_steps to 1 over the warm-up steps and then
    follows half a cosine down to 1 / min_lr_scale at the last training step.
    """
    floor = 1.0 / config.min_lr_scale
    decay_steps = max(1, config.train_steps - config.warmup_steps)

    def factor(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1.0, (step - config.warmup_steps) / decay_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
