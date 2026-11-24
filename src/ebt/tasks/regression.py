"""Two-branch regression y = ±sqrt(x): data, initial candidates and metrics (method.md 16)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ebt.config import RegressionDataConfig
from ebt.core.energy import EnergyModel
from ebt.training.batch import Batch

BRANCH_TOLERANCE = 0.1
"""ε: a prediction is on a branch if its normalized branch distance r is at most ε."""


@dataclass
class RegressionData:
    train: Batch
    validation: Batch
    test: Batch
    ood: Batch
    """Out-of-distribution split for G5; not evaluated for G2."""
    init_std: float
    """Standard deviation of the training targets, used for the initial candidates."""


def _split(size: int, x_min: float, x_max: float, generator: torch.Generator) -> Batch:
    """x ~ U[x_min, x_max]; y = s·sqrt(x) with s = ±1 equally likely and independent of x."""
    x = x_min + (x_max - x_min) * torch.rand(size, 1, generator=generator)
    sign = torch.randint(0, 2, (size, 1), generator=generator).float() * 2 - 1
    return Batch(context=x, target=sign * x.sqrt())


def split_seeds(seed: int) -> dict[str, int]:
    """Seed of the generator of each split for data seed `seed`."""
    return {name: seed * 10 + i for i, name in enumerate(("train", "validation", "test", "ood"))}


def make_data(config: RegressionDataConfig, seed: int) -> RegressionData:
    """All splits, each from its own generator (`split_seeds`), on the CPU in float32."""
    seeds = split_seeds(seed)
    sizes = (config.train_size, config.validation_size, config.test_size)
    splits = [
        _split(size, config.x_min, config.x_max, torch.Generator().manual_seed(seeds[name]))
        for name, size in zip(("train", "validation", "test"), sizes, strict=True)
    ]
    ood = _split(
        config.ood_size,
        config.ood_x_min,
        config.ood_x_max,
        torch.Generator().manual_seed(seeds["ood"]),
    )
    return RegressionData(*splits, ood=ood, init_std=splits[0].target.std().item())


@dataclass(frozen=True)
class Standardization:
    """Means and standard deviations of x and y on the training split (method.md 16.8)."""

    x_mean: float
    x_std: float
    y_mean: float
    y_std: float

    @classmethod
    def of(cls, train: Batch) -> "Standardization":
        x, y = train.context, train.target
        return cls(x.mean().item(), x.std().item(), y.mean().item(), y.std().item())

    def apply(self, batch: Batch) -> Batch:
        """The batch with x and y mapped to zero mean and unit variance."""
        return Batch(
            (batch.context - self.x_mean) / self.x_std, (batch.target - self.y_mean) / self.y_std
        )

    def target_to_original(self, y: Tensor) -> Tensor:
        return y * self.y_std + self.y_mean


def initial_candidates(batch: Batch, init_std: float, generator: torch.Generator) -> Tensor:
    """ŷ_0 ~ N(0, init_std²), drawn independently of the target (method.md section 16)."""
    return init_std * torch.randn(
        batch.target.shape,
        generator=generator,
        dtype=batch.target.dtype,
        device=batch.target.device,
    )


def branch_distance(x: Tensor, prediction: Tensor) -> Tensor:
    """r = min(|ŷ − √x|, |ŷ + √x|) / √x: 0 on a branch, 1 at the conditional mean 0."""
    root = x.sqrt()
    return torch.minimum((prediction - root).abs(), (prediction + root).abs()) / root


def regression_metrics(data: Batch, prediction: Tensor, initial: Tensor | None) -> dict[str, float]:
    """The G2 metrics of one evaluation split (questions.md G2, method.md section 16).

    - branch_distance: mean of r;
    - on_branch: fraction with r <= ε;
    - positive_branch: fraction of predictions above 0;
    - positive_branch_on_branch: the same among predictions with r <= ε (NaN if there are none);
    - mse: mean squared error against the target;
    - mean_predictor_mse: mean of y² on this split, the MSE of always predicting 0;
    - for an EBT also initial_sign_agreement, P(sign ŷ_final = sign ŷ_0), and
      mean_move, the mean of |ŷ_final − ŷ_0|.
    """
    x, target = data.context.cpu(), data.target.cpu()
    distance = branch_distance(x, prediction)
    on_branch = distance <= BRANCH_TOLERANCE
    positive = prediction > 0
    metrics = {
        "branch_distance": distance.mean().item(),
        "on_branch": on_branch.float().mean().item(),
        "positive_branch": positive.float().mean().item(),
        "positive_branch_on_branch": positive[on_branch].float().mean().item()
        if on_branch.any()
        else float("nan"),
        "mse": ((prediction - target) ** 2).mean().item(),
        "mean_predictor_mse": (target**2).mean().item(),
    }
    if initial is not None:
        same_sign = torch.sign(prediction) == torch.sign(initial)
        metrics["initial_sign_agreement"] = same_sign.float().mean().item()
        metrics["mean_move"] = (prediction - initial).abs().mean().item()
    return metrics


def energy_grid(model: EnergyModel, x: Tensor, y: Tensor) -> Tensor:
    """E(x_i, y_j) for every pair, shape (len(x), len(y)), on the CPU."""
    device = next(model.parameters()).device
    context = x.repeat_interleave(len(y)).unsqueeze(-1).to(device)
    candidate = y.repeat(len(x)).unsqueeze(-1).to(device)
    with torch.no_grad():
        energy = model.energy(context, model.encode_candidate(candidate))
    return energy.reshape(len(x), len(y)).cpu()


def energy_gradient_magnitude(model: EnergyModel, x: Tensor, candidate: Tensor) -> float:
    """Mean of |∂E(x, ŷ)/∂ŷ| over the rows, at the given candidates (method.md section 16.7).

    Evaluated at the test targets (on the branches) and at ŷ = 0, it measures the energy scale
    that a thinking step of size α multiplies.
    """
    with torch.enable_grad():
        point = candidate.detach().clone().requires_grad_(True)
        energy = model.energy(x, model.encode_candidate(point))
        (gradient,) = torch.autograd.grad(energy.sum(), point)
    return gradient.abs().mean().item()
