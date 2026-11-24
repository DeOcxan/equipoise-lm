"""Configuration dataclasses. Names follow docs/method.md section 15."""

from dataclasses import dataclass
from enum import StrEnum


class TrainingMode(StrEnum):
    """Gradient flow through the thinking steps during training (method.md section 6.2)."""

    S1 = "s1"
    """Detach between steps, second-order graph at every step, loss averaged over steps."""
    S2 = "s2"
    """No detach, second-order graph at the final step only, loss at the final step."""
    FULL = "full"
    """No detach, second-order graph at every step, loss at the final step."""


@dataclass(frozen=True)
class ThinkingConfig:
    """Training-time thinking (method.md sections 6.2, 7, 8 and 10).

    The defaults are neutral placeholders; every task sets its own values.
    """

    mode: TrainingMode = TrainingMode.S1
    step_size: float = 1.0
    step_size_learnable: bool = False
    step_size_jitter: float = 1.0
    """r in method.md section 8.1; 1.0 disables step-size randomization."""
    energy_scaled_step: bool = False
    """Multiply the step size by exp(E / 9) per energy element (method.md section 10)."""
    steps_min: int = 1
    steps_max: int = 1
    """The training step count is drawn uniformly from {steps_min, ..., steps_max}
    (method.md section 8.2)."""
    langevin_std: float = 0.0

    def __post_init__(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.step_size_jitter < 1:
            raise ValueError("step_size_jitter must be at least 1")
        if not 1 <= self.steps_min <= self.steps_max:
            raise ValueError("steps must satisfy 1 <= steps_min <= steps_max")
        if self.langevin_std < 0:
            raise ValueError("langevin_std must be non-negative")


@dataclass(frozen=True)
class InferenceConfig:
    """Evaluation-time thinking (method.md sections 5.2 and 7).

    No step-size or step-count randomness.
    """

    steps: int = 1
    samples: int = 1
    """M, the number of candidates in best-of-M selection."""
    langevin_std: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.samples < 1:
            raise ValueError("samples must be at least 1")
        if self.langevin_std < 0:
            raise ValueError("langevin_std must be non-negative")


@dataclass(frozen=True)
class ReplayConfig:
    """Replay buffer (method.md section 9). Defaults are the upstream code defaults."""

    capacity: int = 192
    fraction: float = 0.5
    """Replayed rows per batch, as a fraction of the fresh batch size."""

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if self.fraction < 0:
            raise ValueError("fraction must be non-negative")


class Tier(StrEnum):
    """Experiment size. The fast tier runs in tests and CI; the full tier produces saved results."""

    FAST = "fast"
    FULL = "full"


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, schedule and batching shared by every task (method.md sections 10 and 16).

    Defaults are the authors' values (Table D.2 of the paper).
    """

    learning_rate: float
    train_steps: int
    batch_size: int
    warmup_steps: int
    weight_decay: float = 0.01
    min_lr_scale: float = 10.0
    """The cosine schedule decays the learning rate to learning_rate / min_lr_scale."""
    gradient_clip: float = 1.0
    step_size_lr_multiplier: float = 1.0
    """Learning rate of a learnable α relative to learning_rate (method.md section 10)."""

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.train_steps < 1 or self.batch_size < 1:
            raise ValueError("train_steps and batch_size must be at least 1")
        if not 0 <= self.warmup_steps <= self.train_steps:
            raise ValueError("warmup_steps must lie in [0, train_steps]")
        if self.min_lr_scale < 1:
            raise ValueError("min_lr_scale must be at least 1")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights of the EBT training loss terms (method.md sections 6.2 and 16.4)."""

    reconstruction_weight: float = 1.0
    """Weight of the loss of the thinking predictions against the target (section 6.2)."""
    contrastive_weight: float = 0.0
    """Weight of the contrastive energy loss (section 16.4)."""
    energy_regularization_weight: float = 0.0
    """λ of the energy-magnitude penalty λ · mean(E_real² + E_fake²) (section 16.4)."""

    def __post_init__(self) -> None:
        weights = (
            self.reconstruction_weight,
            self.contrastive_weight,
            self.energy_regularization_weight,
        )
        if min(weights) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.reconstruction_weight == 0 and self.contrastive_weight == 0:
            raise ValueError("at least one loss weight must be positive")


ACTIVATIONS = ("silu", "relu", "tanh")


@dataclass(frozen=True)
class MLPConfig:
    """Hidden layers of an MLP energy, regressor or mixture density network (method.md 16.2).

    The activation is SiLU for the EBTs and the regressor, ReLU for the implicit behavioural
    cloning energy and tanh for the mixture density network, as in their sources (section 16.8).
    """

    widths: tuple[int, ...] = (64, 64)
    activation: str = "silu"

    def __post_init__(self) -> None:
        if not self.widths or min(self.widths) < 1:
            raise ValueError("an MLP needs at least one hidden layer of positive width")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")


@dataclass(frozen=True)
class RegressionDataConfig:
    """Two-branch regression data y = ±sqrt(x) (method.md section 16)."""

    train_size: int
    validation_size: int
    test_size: int
    ood_size: int
    x_min: float = 0.25
    x_max: float = 4.0
    ood_x_min: float = 4.0
    ood_x_max: float = 9.0
    """Out-of-distribution range for G5; generated but not evaluated for G2."""

    def __post_init__(self) -> None:
        if not 0 < self.x_min < self.x_max <= self.ood_x_min < self.ood_x_max:
            raise ValueError("ranges must satisfy 0 < x_min < x_max <= ood_x_min < ood_x_max")


@dataclass(frozen=True)
class PatchTransformerConfig:
    """Size of the patch-transformer energy and denoiser (method.md section 17.3)."""

    image_size: int = 28
    channels: int = 1
    patch: int = 4
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 1.0
    """Hidden width of each block's MLP relative to dim; 1 in the authors' image scripts."""

    def __post_init__(self) -> None:
        if self.image_size % self.patch:
            raise ValueError("the patch size must divide the image size")
        if self.dim % self.heads or (self.dim // 2) % 2:
            raise ValueError("dim must be divisible by heads, and dim / 2 must be even")
