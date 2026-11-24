"""MNIST denoising: an S1 EBT against a feed-forward denoiser of the same size (method.md 17).

    python -u -m ebt.experiments.mnist_denoising --tier full --device mps --stage all

Hyperparameters are chosen on validation data by each model's own objective (method.md
section 17.5); the test split is used only by the final runs. Every search configuration and
every final run is written to disk as soon as it finishes, and a rerun skips the work already
on disk.
"""

import argparse
import dataclasses
import itertools
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebt.checkpoints import checkpoint_directory, load_checkpoint
from ebt.config import (
    InferenceConfig,
    ObjectiveConfig,
    PatchTransformerConfig,
    ThinkingConfig,
    Tier,
    TrainingConfig,
    TrainingMode,
)
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import runner
from ebt.fingerprint import code_fingerprint
from ebt.models.mlp import parameter_count
from ebt.models.patch_transformer import PatchDenoiser, PatchEnergy
from ebt.results import RunResult, environment
from ebt.runtime import make_generator, run_seed, seed_everything, select_device
from ebt.tasks.mnist import (
    ImageData,
    add_noise,
    alpha_bar,
    denoising_metrics,
    load_images,
    make_data,
)
from ebt.training.batch import Batch
from ebt.training.evaluation import predict_ebt, predict_feedforward
from ebt.training.objectives import EBTObjective, FeedForwardObjective
from ebt.training.trainer import train

EXPERIMENT = "mnist_denoising"
MODELS = ("ebt", "denoiser")
DATA_SEED = 0
SEARCH_SEED = 100
REPORT_SEEDS = (0, 1, 2, 3, 4)
VALIDATION_SIZE = 5_000
NOISE_LEVEL = 0.1
VALIDATION_NOISE_SEED, TEST_NOISE_SEED = 40_000, 40_001
"""Seeds of the fixed noise of the validation and test images, the same for every run."""
FIXED_SEEDS = {
    "data_split": DATA_SEED,
    "validation_noise": VALIDATION_NOISE_SEED,
    "test_noise": TEST_NOISE_SEED,
}
"""Seeds shared by every run on purpose: the same split and the same noisy evaluation images."""
RUN_SOURCES = ("initialization", "batch", "noise")
"""The random sources of one run, each with its own seed (`run_seeds`)."""


def run_seeds(seed: int) -> dict[str, int]:
    """Seed of each random source of run `seed`: `run_seed` of the source's index, disjoint
    across sources and runs and from FIXED_SEEDS."""
    return {name: run_seed(index, seed) for index, name in enumerate(RUN_SOURCES)}


EXAMPLES = 8
"""Test images whose denoising is saved for the plots."""
INFERENCE = InferenceConfig(steps=2)
"""Two inference steps, as in the authors' denoising inference script (method.md 17.5)."""
STEP_SIZE_LR_FACTOR = 3.0
"""The learning-rate multiplier of α is 3 × its initial value (§I.2; upstream 3000 and 9000)."""


def smooth_l1(prediction: Tensor, target: Tensor) -> Tensor:
    """Smooth L1 with β = 1, the authors' image loss (method.md section 6.1)."""
    return F.smooth_l1_loss(prediction, target, beta=1.0)


@dataclass(frozen=True)
class TierSettings:
    model: PatchTransformerConfig
    train_steps: int
    batch_size: int
    warmup_steps: int
    validate_every: int
    eval_batch: int


TIERS = {
    Tier.FAST: TierSettings(
        PatchTransformerConfig(dim=32, depth=1, heads=2),
        train_steps=40,
        batch_size=16,
        warmup_steps=4,
        validate_every=20,
        eval_batch=64,
    ),
    Tier.FULL: TierSettings(
        PatchTransformerConfig(patch=4, dim=128, depth=4, heads=4),
        train_steps=10_000,
        batch_size=128,
        warmup_steps=1_000,
        validate_every=500,
        eval_batch=1_000,
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One hyperparameter configuration; `step_size` is the initial α of the EBT."""

    model: str
    learning_rate: float
    step_size: float | None = None


def search_space(model: str, tier: Tier) -> list[Candidate]:
    """The configurations tried on validation data, in grid order (method.md section 17.5).

    The first grid is followed by its extension at the edge where both first selections lay:
    learning rate 3e-3 for both models and, for the EBT, α = 100 000.
    """
    if tier is Tier.FAST:
        return [Candidate(model, 1e-3, 3000.0 if model == "ebt" else None)]
    if model == "denoiser":
        return [Candidate(model, lr) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]
    first = itertools.product((1e-4, 3e-4, 1e-3), (300.0, 3000.0, 30000.0))
    extension = ((3e-3, 30000.0), (3e-3, 100000.0), (1e-3, 100000.0))
    return [Candidate(model, lr, alpha) for lr, alpha in (*first, *extension)]


def thinking_config(candidate: Candidate) -> ThinkingConfig:
    return ThinkingConfig(
        mode=TrainingMode.S1, step_size=candidate.step_size, step_size_learnable=True
    )


def training_config(candidate: Candidate, settings: TierSettings) -> TrainingConfig:
    multiplier = 1.0 if candidate.step_size is None else STEP_SIZE_LR_FACTOR * candidate.step_size
    return TrainingConfig(
        learning_rate=candidate.learning_rate,
        train_steps=settings.train_steps,
        batch_size=settings.batch_size,
        warmup_steps=settings.warmup_steps,
        step_size_lr_multiplier=multiplier,
    )


def fixed_noise(images: Tensor, seed: int) -> Tensor:
    """Noisy copies of `images` from a CPU generator with `seed`, identical in every run."""
    return add_noise(images, alpha_bar(NOISE_LEVEL), torch.Generator().manual_seed(seed))


def run(
    candidate: Candidate,
    tier: Tier,
    seed: int,
    device: torch.device,
    with_test: bool,
    data: ImageData,
) -> tuple[RunResult, tuple[str, float], nn.Module]:
    """Train one model with one seed; evaluate on validation and, if requested, test data."""
    settings = TIERS[tier]
    seeds = run_seeds(seed)
    seed_everything(seeds["initialization"])
    alpha = alpha_bar(NOISE_LEVEL)
    batch_generator = torch.Generator().manual_seed(seeds["batch"])
    noise_generator = make_generator(seeds["noise"], device)
    training = training_config(candidate, settings)

    def noised(batch: Batch, generator: torch.Generator) -> Tensor:
        return add_noise(batch.target, alpha, generator)

    loop = None
    if candidate.model == "ebt":
        model = PatchEnergy(settings.model).to(device)
        thinking = thinking_config(candidate)
        loop = ThinkingLoop(thinking).to(device)
        objective = EBTObjective(model, loop, smooth_l1, ObjectiveConfig(), initial=noised)
    else:
        model = PatchDenoiser(settings.model).to(device)
        thinking = None
        objective = FeedForwardObjective(model, smooth_l1, inputs=noised)

    def denoise(noisy: Tensor):
        if loop is None:
            return predict_feedforward(model, noisy.to(device), settings.eval_batch)
        batch = Batch(None, noisy.to(device))
        return predict_ebt(
            model, loop, batch, batch.target, INFERENCE, None, settings.eval_batch, False
        )

    clean = {"validation": data.validation.target}
    noisy = {"validation": fixed_noise(clean["validation"], VALIDATION_NOISE_SEED)}
    if with_test:
        clean["test"] = data.test.target
        noisy["test"] = fixed_noise(clean["test"], TEST_NOISE_SEED)

    def validate() -> dict[str, float]:
        metrics = denoising_metrics(denoise(noisy["validation"]).final, clean["validation"])
        return {"smooth_l1": metrics["smooth_l1"], "psnr": metrics["psnr"]}

    history = train(
        objective,
        data.train.to(device),
        training,
        batch_generator,
        noise_generator,
        validate=validate,
        validate_every=settings.validate_every,
    )

    predictions = {name: denoise(images) for name, images in noisy.items()}
    metrics = {
        name: {
            **denoising_metrics(predictions[name].final, clean[name]),
            "noisy_input_psnr": denoising_metrics(noisy[name], clean[name])["psnr"],
        }
        for name in noisy
    }
    selection = ("validation_smooth_l1", metrics["validation"]["smooth_l1"])
    reported = predictions["test" if with_test else "validation"]
    step_size = None if loop is None else loop.step_size.raw.clamp(min=1e-4).item()
    result = RunResult(
        experiment=EXPERIMENT,
        task="mnist_denoising",
        model=candidate.model,
        tier=tier.value,
        seed=seed,
        device=device.type,
        dtype="float32",
        environment=environment(),
        config={
            "candidate": dataclasses.asdict(candidate),
            "model": dataclasses.asdict(settings.model),
            "training": dataclasses.asdict(training),
            "thinking": dataclasses.asdict(thinking) if thinking else None,
            "inference": dataclasses.asdict(INFERENCE) if loop else None,
            "noise_level": NOISE_LEVEL,
            "alpha_bar": alpha,
            "data_seed": DATA_SEED,
            "run_seeds": seeds,
            "validation_size": VALIDATION_SIZE,
        },
        parameters={"trainable": parameter_count(objective.module)},
        metrics=metrics,
        cost={
            "train_total_example_passes": dataclasses.asdict(history.cost),
            "train_examples": history.examples,
            "inference_example_passes_per_example": {
                key: value / len(reported.final)
                for key, value in dataclasses.asdict(reported.cost).items()
            },
        },
        history={
            "steps": history.steps,
            "train_loss": history.train_loss,
            "validation_steps": history.validation_steps,
            **{f"validation_{name}": values for name, values in history.validation.items()},
            "final_step_size": step_size,
        },
    )
    if with_test and seed == REPORT_SEEDS[0]:
        result.artifacts = examples(model, loop, noisy["test"], clean["test"], device)
    return result, selection, objective.module


def restore(path: Path, device: torch.device) -> tuple[nn.Module, ThinkingLoop | None]:
    """The trained model of a final run and, for the EBT, its thinking loop (with α), loaded
    from the run's checkpoint."""
    saved = load_checkpoint(path)
    candidate = Candidate(**saved["run"]["config"]["candidate"])
    settings = TIERS[Tier(saved["run"]["tier"])]
    if candidate.model == "denoiser":
        model = PatchDenoiser(settings.model).to(device)
        model.load_state_dict(saved["state_dict"])
        return model, None
    model = PatchEnergy(settings.model).to(device)
    loop = ThinkingLoop(thinking_config(candidate)).to(device)
    nn.ModuleList([model, loop]).load_state_dict(saved["state_dict"])
    return model, loop


def examples(model, loop, noisy: Tensor, clean: Tensor, device: torch.device) -> dict:
    """The first EXAMPLES test images: clean, noisy and the output after every step."""
    rows = slice(0, EXAMPLES)
    if loop is None:
        with torch.no_grad():
            steps = [model(noisy[rows].to(device)).cpu()]
    else:
        trajectory = loop.infer(model, None, noisy[rows].to(device), INFERENCE)
        steps = [candidate.cpu() for candidate in trajectory.candidates[1:]]
    return {
        "examples": {
            "clean": clean[rows].squeeze(1).tolist(),
            "noisy": noisy[rows].squeeze(1).tolist(),
            "outputs": [output.squeeze(1).tolist() for output in steps],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--stage", choices=["search", "final", "all"], default="all")
    parser.add_argument("--data", type=Path, default=Path("data/mnist/raw"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--reuse-stale", action="store_true", help=runner.REUSE_HELP)
    args = parser.parse_args()
    tier, device = Tier(args.tier), select_device(args.device)
    data = make_data(*load_images(args.data), VALIDATION_SIZE, DATA_SEED)
    runner.execute(
        args.output or Path("results/mnist/denoising") / tier.value,
        args.stage,
        tuple(args.models),
        lambda model: search_space(model, tier),
        Candidate,
        partial(_train_one, tier=tier, device=device, data=data),
        SEARCH_SEED,
        REPORT_SEEDS,
        runner.Provenance(code_fingerprint("mnist_denoising"), args.reuse_stale),
        checkpoint_directory(EXPERIMENT, tier.value),
    )


def _train_one(candidate, seed, with_test, *, tier, device, data):
    return run(candidate, tier, seed, device, with_test, data)


if __name__ == "__main__":
    main()
