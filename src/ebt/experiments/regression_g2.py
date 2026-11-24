"""Question G2 on the two-branch regression task (questions.md G2, method.md section 16).

Eight models: (a) an MLP regressor, (a2) a mixture density network, (b) an EBT trained with MSE
on the final candidate, (c1) an EBT trained with the contrastive loss alone, (c2) an EBT
trained with MSE plus the contrastive loss, (c3) an EBT trained with the contrastive loss plus
an energy-magnitude penalty, (c4) an implicit behavioural cloning energy and (c5) an IRED
energy. Hyperparameters are chosen on validation data by each model's own training objective
(method.md section 16.6); the test split is used only by the final runs.

    python -u -m ebt.experiments.regression_g2 --tier full --device cpu --stage all

Every search configuration and every final run is written to disk as soon as it finishes, and a
rerun skips the work already on disk.
"""

import argparse
import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebt.checkpoints import checkpoint_directory, load_checkpoint
from ebt.config import (
    InferenceConfig,
    MLPConfig,
    ObjectiveConfig,
    RegressionDataConfig,
    ReplayConfig,
    ThinkingConfig,
    Tier,
    TrainingConfig,
    TrainingMode,
)
from ebt.core.contrastive import contrastive_loss, energy_magnitude
from ebt.core.cost import PassCount
from ebt.core.energy import EnergyModel
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import runner
from ebt.fingerprint import code_fingerprint
from ebt.inference.annealed import LandscapeSteps, annealed_sample
from ebt.inference.dfo import DFOConfig, derivative_free_optimization
from ebt.models.ired import LandscapeEnergy, OnLandscape, cosine_schedule
from ebt.models.mdn import (
    Mixture,
    MixtureDensityNetwork,
    log_density,
    most_probable_branch,
    most_probable_value,
    negative_log_likelihood,
    sampled_centre,
)
from ebt.models.mlp import MLPEnergy, MLPRegressor, parameter_count
from ebt.models.standardized import StandardizedEnergy
from ebt.results import RunResult, environment
from ebt.runtime import make_generator, run_seed, seed_everything, select_device
from ebt.tasks.regression import (
    Standardization,
    energy_gradient_magnitude,
    energy_grid,
    initial_candidates,
    make_data,
    regression_metrics,
    split_seeds,
)
from ebt.training.batch import Batch
from ebt.training.evaluation import Predictions, predict_ebt, predict_feedforward
from ebt.training.ired import IREDObjective
from ebt.training.objectives import (
    EBTObjective,
    FeedForwardObjective,
    InfoNCEObjective,
    TrainingObjective,
)
from ebt.training.trainer import train

EXPERIMENT = "regression_g2"
MODELS = (
    "regressor",
    "mdn",
    "ebt_mse",
    "ebt_contrastive",
    "ebt_combined",
    "ebt_regularized",
    "ibc",
    "ired",
)
EBT_MODELS = ("ebt_mse", "ebt_contrastive", "ebt_combined", "ebt_regularized")
DATA_SEED = 0
SEARCH_SEED = 100
REPORT_SEEDS = (0, 1, 2, 3, 4)
SELECTION_COUNTER_SEED = 20_000
"""Seed of the fixed counter-examples, noise and landscapes of the IBC and IRED selection
objectives, the same for every configuration."""
FIXED_SEEDS = {
    **{f"data_{name}": value for name, value in split_seeds(DATA_SEED).items()},
    "selection_counter_examples": SELECTION_COUNTER_SEED,
}
"""Seeds shared by every run on purpose: the same data and the same validation objective."""
RUN_SOURCES = (
    "initialization",
    "batch",
    "noise",
    "replay",
    "evaluation",
    "mixture_readout",
)
"""The random sources of one run, each with its own seed (`run_seeds`)."""


def run_seeds(seed: int) -> dict[str, int]:
    """Seed of each random source of run `seed` (method.md section 16.6): `run_seed` of the
    source's index, disjoint across sources and runs and from FIXED_SEEDS."""
    return {name: run_seed(index, seed) for index, name in enumerate(RUN_SOURCES)}


EVAL_BATCH = 4096
DFO_BATCH = 64
"""Rows per call of the derivative-free optimizer, which evaluates 16 384 candidates per row."""
ARTIFACT_ROWS = 512
GRID_X = torch.linspace(0.25, 4.0, 61)
GRID_Y = torch.linspace(-2.5, 2.5, 101)

ENERGY_MODEL = MLPConfig(widths=(64, 64))
"""4416 trainable parameters: (2·64 + 64) + (64·64 + 64) + 64."""
REGRESSOR_MODEL = MLPConfig(widths=(65, 64))
"""4419 trainable parameters: (65 + 65) + (65·64 + 64) + (64 + 1), within 0.1% of the energy."""
IBC_MODEL = MLPConfig(widths=(64, 64), activation="relu")
"""4416 trainable parameters, as ENERGY_MODEL, with ReLU as in Florence et al. (2021)."""
MDN_WIDTHS = {2: (49, 77), 5: (71, 49), 10: (47, 55)}
"""Hidden widths per number of kernels m, chosen for 4416, 4420 and 4414 trainable parameters:
(49 + 49) + (49·77 + 77) + (77·6 + 6), (71 + 71) + (71·49 + 49) + (49·15 + 15) and
(47 + 47) + (47·55 + 55) + (55·30 + 30)."""
IRED_SIZES = {"hidden": 37, "embedding": 8, "time": 8}
"""4437 trainable parameters, 0.5% above the energy (method.md section 16.8)."""
IRED_LANDSCAPES = 10
IBC_COUNTER_EXAMPLES = 256
IBC_BOUNDARY_BUFFER = 0.05
"""Counter-examples and DFO samples lie in [min, max] of the standardized training targets,
widened by 5% of their range on each side."""
STEPS_MIN, STEPS_MAX = 2, 4
"""Upstream text S2 step range, k ∈ {2, 3, 4} (method.md D9)."""
STEP_SIZE_JITTER = 2.0
REPLAY = ReplayConfig(capacity=192, fraction=0.5)
INFERENCE = InferenceConfig(steps=STEPS_MAX, samples=1, langevin_std=0.0)


@dataclass(frozen=True)
class TierSettings:
    data: RegressionDataConfig
    train_steps: int
    batch_size: int
    warmup_steps: int
    dfo: DFOConfig


TIERS = {
    Tier.FAST: TierSettings(
        RegressionDataConfig(train_size=512, validation_size=256, test_size=256, ood_size=256),
        train_steps=200,
        batch_size=64,
        warmup_steps=20,
        dfo=DFOConfig(samples=256),
    ),
    Tier.FULL: TierSettings(
        RegressionDataConfig(train_size=8192, validation_size=2048, test_size=4096, ood_size=4096),
        train_steps=20_000,
        batch_size=256,
        warmup_steps=1_000,
        dfo=DFOConfig(),
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One hyperparameter configuration of one model; None where a value does not apply."""

    model: str
    learning_rate: float
    step_size: float | None = None
    langevin_std: float | None = None
    contrastive_weight: float | None = None
    energy_regularization: float | None = None
    components: int | None = None
    """m, the number of kernels of the mixture density network."""


def search_space(model: str, tier: Tier) -> list[Candidate]:
    """The configurations tried on validation data (method.md sections 16.6 and 16.8).

    The fast tier tries one configuration per model, to exercise the code.
    """
    if tier is Tier.FAST:
        if model in ("regressor", "ibc", "ired"):
            return [Candidate(model, 3e-3)]
        if model == "mdn":
            return [Candidate(model, 3e-3, components=2)]
        weights = {"ebt_contrastive": 1.0, "ebt_combined": 0.05, "ebt_regularized": 1.0}
        regularization = 0.1 if model == "ebt_regularized" else None
        return [Candidate(model, 3e-3, 0.3, 0.1, weights.get(model), regularization)]
    if model == "regressor":
        return [Candidate(model, lr) for lr in (1e-3, 3e-3, 1e-2)]
    if model == "mdn":
        return [
            Candidate(model, lr, components=m)
            for lr, m in itertools.product((1e-3, 3e-3, 1e-2), (2, 5, 10))
        ]
    if model == "ibc":
        return [Candidate(model, lr) for lr in (3e-4, 1e-3, 3e-3)]
    if model == "ired":
        return [Candidate(model, lr) for lr in (1e-4, 3e-4, 1e-3)]
    grid = itertools.product((1e-3, 3e-3), (0.1, 0.3, 1.0), (0.0, 0.1, 0.3))
    weights = {
        "ebt_mse": (None,),
        "ebt_contrastive": (1.0,),
        "ebt_combined": (0.0005, 0.05, 1.0),
        "ebt_regularized": (1.0,),
    }
    regularizations = (0.01, 0.1, 1.0) if model == "ebt_regularized" else (None,)
    return [
        Candidate(model, lr, alpha, sigma, weight, regularization)
        for (lr, alpha, sigma), weight, regularization in itertools.product(
            grid, weights[model], regularizations
        )
    ]


def objective_weights(candidate: Candidate) -> ObjectiveConfig:
    """Reconstruction and contrastive weights of each EBT model (method.md section 16.3)."""
    if candidate.model == "ebt_mse":
        return ObjectiveConfig(reconstruction_weight=1.0, contrastive_weight=0.0)
    if candidate.model == "ebt_contrastive":
        return ObjectiveConfig(reconstruction_weight=0.0, contrastive_weight=1.0)
    if candidate.model == "ebt_regularized":
        return ObjectiveConfig(
            reconstruction_weight=0.0,
            contrastive_weight=1.0,
            energy_regularization_weight=candidate.energy_regularization,
        )
    return ObjectiveConfig(
        reconstruction_weight=1.0, contrastive_weight=candidate.contrastive_weight
    )


def thinking_config(candidate: Candidate) -> ThinkingConfig:
    return ThinkingConfig(
        mode=TrainingMode.S2,
        step_size=candidate.step_size,
        step_size_jitter=STEP_SIZE_JITTER,
        steps_min=STEPS_MIN,
        steps_max=STEPS_MAX,
        langevin_std=candidate.langevin_std,
    )


def training_config(candidate: Candidate, settings: TierSettings) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=candidate.learning_rate,
        train_steps=settings.train_steps,
        batch_size=settings.batch_size,
        warmup_steps=settings.warmup_steps,
    )


@dataclass
class Setup:
    """What one model needs for training, evaluation and its result file."""

    objective: TrainingObjective
    train_data: Batch
    """Training rows in the units the model is trained in."""
    predict: Callable[[Batch, torch.Generator], Predictions]
    """Predictions for an evaluation split in the original units."""
    select: Callable[[Batch, Predictions], tuple[str, float]]
    """The model's own training objective on validation data (method.md section 16.6)."""
    energy: EnergyModel | None
    """Energy in the original units for the energy grid and the gradient diagnostic."""
    step_size: float | None
    """Step size, in the original units, that multiplies the gradient in the diagnostic."""
    config: dict
    trained: nn.Module
    """Every trained parameter of the model, saved in the checkpoint of a final run."""
    extra_metrics: Callable[[str, Batch], dict] = lambda name, split: {}
    """Further metrics of a split, keyed by a name that includes the split name."""
    extra_artifacts: dict = field(default_factory=dict)


def selection_objective(
    candidate: Candidate, model, data: Batch, predictions: Predictions
) -> tuple[str, float]:
    """Each EBT's or the regressor's own training objective on validation data (16.6).

    MSE for the regressor and the MSE EBT; the contrastive loss for the contrastive EBT; MSE
    plus the weighted contrastive loss for the combined EBT; the contrastive loss plus
    λ · mean(E_real² + E_fake²) for the regularized EBT. The EBT terms are evaluated on
    the inference trajectory: MSE of the final candidate, and the contrastive loss with the
    energy at the input of the final step as the fake energy.
    """
    mse = F.mse_loss(predictions.final, data.target.cpu()).item()
    if candidate.model in ("regressor", "ebt_mse"):
        return "validation_mse", mse
    with torch.no_grad():
        real = model.energy(data.context, model.encode_candidate(data.target)).cpu()
    fake = predictions.last_step_energy
    contrastive = contrastive_loss(real, fake).item()
    if candidate.model == "ebt_contrastive":
        return "validation_contrastive", contrastive
    if candidate.model == "ebt_regularized":
        penalty = energy_magnitude(real, fake).item()
        return (
            "validation_contrastive_regularized",
            contrastive + candidate.energy_regularization * penalty,
        )
    return "validation_combined", mse + candidate.contrastive_weight * contrastive


def regressor_setup(candidate, data, train_data, device, seed) -> Setup:
    model = MLPRegressor(1, 1, REGRESSOR_MODEL).to(device)
    return Setup(
        objective=FeedForwardObjective(model, F.mse_loss),
        train_data=train_data,
        predict=lambda split, generator: predict_feedforward(model, split.context, EVAL_BATCH),
        select=partial(selection_objective, candidate, model),
        energy=None,
        step_size=None,
        config={"model": dataclasses.asdict(REGRESSOR_MODEL)},
        trained=model,
    )


def ebt_setup(candidate, data, train_data, device, seed) -> Setup:
    model = MLPEnergy(1, 1, ENERGY_MODEL).to(device)
    thinking = thinking_config(candidate)
    loop = ThinkingLoop(thinking).to(device)
    objective_config = objective_weights(candidate)
    objective = EBTObjective(
        model,
        loop,
        F.mse_loss,
        objective_config,
        initial=lambda batch, generator: initial_candidates(batch, data.init_std, generator),
        replay=ReplayBuffer(REPLAY, torch.Generator().manual_seed(run_seeds(seed)["replay"])),
    )

    def predict(split: Batch, generator: torch.Generator) -> Predictions:
        initial = initial_candidates(split, data.init_std, generator)
        return predict_ebt(model, loop, split, initial, INFERENCE, generator, EVAL_BATCH)

    return Setup(
        objective=objective,
        train_data=train_data,
        predict=predict,
        select=partial(selection_objective, candidate, model),
        energy=model,
        step_size=loop.step_size.raw.clamp(min=1e-4).item(),
        config={
            "model": dataclasses.asdict(ENERGY_MODEL),
            "thinking": dataclasses.asdict(thinking),
            "inference": dataclasses.asdict(INFERENCE),
            "objective": dataclasses.asdict(objective_config),
            "replay": dataclasses.asdict(REPLAY),
        },
        trained=objective.module,
    )


def _mixture(model: MixtureDensityNetwork, context: Tensor) -> Mixture:
    with torch.no_grad():
        return model(context)


def mdn_setup(candidate, data, train_data, device, seed) -> Setup:
    config = MLPConfig(widths=MDN_WIDTHS[candidate.components], activation="tanh")
    model = MixtureDensityNetwork(1, candidate.components, config).to(device)
    readout_generator = make_generator(run_seeds(seed)["mixture_readout"], device)

    def predict(split: Batch, generator: torch.Generator) -> Predictions:
        final = most_probable_branch(_mixture(model, split.context)).cpu()
        return Predictions(None, final, None, None, PassCount(forward=len(split)))

    def select(split: Batch, predictions: Predictions) -> tuple[str, float]:
        mixture = _mixture(model, split.context)
        return "validation_nll", negative_log_likelihood(mixture, split.target).item()

    def extra_metrics(name: str, split: Batch) -> dict:
        """Metrics of the most probable value (eq. 48) and of a sampled kernel centre."""
        mixture = _mixture(model, split.context)
        sampled = sampled_centre(mixture, readout_generator)
        return {
            f"{name}_most_probable_value": regression_metrics(
                split, most_probable_value(mixture).cpu(), None
            ),
            f"{name}_sampled_centre": regression_metrics(split, sampled.cpu(), None),
            f"{name}_nll": {"nll": negative_log_likelihood(mixture, split.target).item()},
        }

    grid_context = GRID_X.repeat_interleave(len(GRID_Y)).unsqueeze(-1).to(device)
    grid_target = GRID_Y.repeat(len(GRID_X)).unsqueeze(-1).to(device)
    return Setup(
        objective=FeedForwardObjective(model, negative_log_likelihood),
        train_data=train_data,
        predict=predict,
        select=select,
        energy=None,
        step_size=None,
        config={"model": dataclasses.asdict(config), "readout": "most probable branch, eq. 49"},
        trained=model,
        extra_metrics=extra_metrics,
        extra_artifacts={
            "density_grid": lambda: {
                "x": GRID_X.tolist(),
                "y": GRID_Y.tolist(),
                "negative_log_density": (-log_density(_mixture(model, grid_context), grid_target))
                .reshape(len(GRID_X), len(GRID_Y))
                .cpu()
                .tolist(),
            }
        },
    )


def _bounds(standardized_train: Batch) -> tuple[float, float]:
    low, high = standardized_train.target.min().item(), standardized_train.target.max().item()
    buffer = IBC_BOUNDARY_BUFFER * (high - low)
    return low - buffer, high + buffer


def ibc_setup(candidate, data, train_data, device, seed, dfo: DFOConfig) -> Setup:
    standardization = Standardization.of(train_data)
    standardized_train = standardization.apply(train_data)
    bounds = _bounds(standardized_train)
    model = MLPEnergy(1, 1, IBC_MODEL).to(device)
    objective = InfoNCEObjective(model, IBC_COUNTER_EXAMPLES, bounds)

    def predict(split: Batch, generator: torch.Generator) -> Predictions:
        context = standardization.apply(split).context
        finals, cost = [], PassCount()
        for start in range(0, len(split), DFO_BATCH):
            chosen, passes = derivative_free_optimization(
                model, context[start : start + DFO_BATCH], bounds, dfo, generator
            )
            finals.append(standardization.target_to_original(chosen).cpu())
            cost = cost + passes
        return Predictions(None, torch.cat(finals), None, None, cost)

    def select(split: Batch, predictions: Predictions) -> tuple[str, float]:
        generator = torch.Generator(device=device).manual_seed(SELECTION_COUNTER_SEED)
        with torch.no_grad():
            loss = objective.batch_loss(standardization.apply(split), generator).loss
        return "validation_info_nce", loss.item()

    return Setup(
        objective=objective,
        train_data=standardized_train,
        predict=predict,
        select=select,
        energy=StandardizedEnergy(model, standardization),
        step_size=None,
        config={
            "model": dataclasses.asdict(IBC_MODEL),
            "standardization": dataclasses.asdict(standardization),
            "counter_examples": IBC_COUNTER_EXAMPLES,
            "bounds_standardized": list(bounds),
            "dfo": dataclasses.asdict(dfo),
        },
        trained=model,
    )


def ired_setup(candidate, data, train_data, device, seed) -> Setup:
    standardization = Standardization.of(train_data)
    model = LandscapeEnergy(1, 1, **IRED_SIZES).to(device)
    steps = LandscapeSteps(cosine_schedule(IRED_LANDSCAPES)).to(device)
    objective = IREDObjective(model, steps)

    def predict(split: Batch, generator: torch.Generator) -> Predictions:
        standardized = standardization.apply(split)
        initial = torch.randn(
            standardized.target.shape, generator=generator, device=standardized.target.device
        )
        finals, cost = [], PassCount()
        for start in range(0, len(split), EVAL_BATCH):
            rows = slice(start, start + EVAL_BATCH)
            final, passes = annealed_sample(model, steps, standardized.context[rows], initial[rows])
            finals.append(final.cpu())
            cost = cost + passes * len(final)
        to_original = standardization.target_to_original
        return Predictions(
            to_original(initial).cpu(), to_original(torch.cat(finals)), None, None, cost
        )

    def select(split: Batch, predictions: Predictions) -> tuple[str, float]:
        generator = torch.Generator(device=device).manual_seed(SELECTION_COUNTER_SEED)
        loss = objective.batch_loss(standardization.apply(split), generator).loss
        return "validation_ired", loss.item()

    schedule = steps.schedule
    smoothest = IRED_LANDSCAPES - 1
    return Setup(
        objective=objective,
        train_data=standardization.apply(train_data),
        predict=predict,
        select=select,
        energy=StandardizedEnergy(OnLandscape(model, 0), standardization),
        step_size=schedule.step_size[0].item() * standardization.y_std**2,
        config={
            "model": IRED_SIZES,
            "standardization": dataclasses.asdict(standardization),
            "landscapes": IRED_LANDSCAPES,
            "alpha_bar": schedule.alpha_bar.tolist(),
            "step_size": schedule.step_size.tolist(),
        },
        trained=nn.ModuleList([model, steps]),
        extra_artifacts={
            "energy_grid_smoothest": lambda: {
                "landscape": smoothest,
                "x": GRID_X.tolist(),
                "y": GRID_Y.tolist(),
                "energy": energy_grid(
                    StandardizedEnergy(OnLandscape(model, smoothest), standardization),
                    GRID_X,
                    GRID_Y,
                ).tolist(),
            }
        },
    )


def setup(candidate, data, train_data, device, seed, settings) -> Setup:
    if candidate.model == "regressor":
        return regressor_setup(candidate, data, train_data, device, seed)
    if candidate.model == "mdn":
        return mdn_setup(candidate, data, train_data, device, seed)
    if candidate.model == "ibc":
        return ibc_setup(candidate, data, train_data, device, seed, settings.dfo)
    if candidate.model == "ired":
        return ired_setup(candidate, data, train_data, device, seed)
    return ebt_setup(candidate, data, train_data, device, seed)


def run(
    candidate: Candidate, tier: Tier, seed: int, device: torch.device, with_test: bool
) -> tuple[RunResult, tuple[str, float], nn.Module]:
    """Train one model with one seed; evaluate on validation and, if requested, test data."""
    settings = TIERS[tier]
    seeds = run_seeds(seed)
    seed_everything(seeds["initialization"])
    data = make_data(settings.data, DATA_SEED)
    train_data, validation = data.train.to(device), data.validation.to(device)
    batch_generator = torch.Generator().manual_seed(seeds["batch"])
    noise_generator = make_generator(seeds["noise"], device)
    evaluation_generator = make_generator(seeds["evaluation"], device)
    training = training_config(candidate, settings)
    model_setup = setup(candidate, data, train_data, device, seed, settings)

    history = train(
        model_setup.objective, model_setup.train_data, training, batch_generator, noise_generator
    )

    splits = {"validation": validation}
    if with_test:
        splits["test"] = data.test.to(device)
    predictions = {
        name: model_setup.predict(split, evaluation_generator) for name, split in splits.items()
    }
    metrics = {
        name: regression_metrics(splits[name], p.final, p.initial)
        for name, p in predictions.items()
    }
    selection = model_setup.select(validation, predictions["validation"])
    metrics["validation"][selection[0]] = selection[1]
    for name, split in splits.items():
        metrics.update(model_setup.extra_metrics(name, split))

    reported = predictions["test" if with_test else "validation"]
    result = RunResult(
        experiment=EXPERIMENT,
        task="regression",
        model=candidate.model,
        tier=tier.value,
        seed=seed,
        device=device.type,
        dtype="float32",
        environment=environment(),
        config={
            "candidate": dataclasses.asdict(candidate),
            "data": dataclasses.asdict(settings.data),
            "data_seed": DATA_SEED,
            "run_seeds": seeds,
            "init_std": data.init_std,
            "training": dataclasses.asdict(training),
            **{key: None for key in ("thinking", "inference", "objective", "replay")},
            **model_setup.config,
        },
        parameters={"trainable": parameter_count(model_setup.objective.module)},
        metrics=metrics,
        cost={
            "train_total_example_passes": dataclasses.asdict(history.cost),
            "train_examples": history.examples,
            "inference_example_passes_per_example": {
                key: value / len(reported.final)
                for key, value in dataclasses.asdict(reported.cost).items()
            },
        },
        history={"steps": history.steps, "train_loss": history.train_loss},
    )
    if with_test and seed == REPORT_SEEDS[0]:
        result.artifacts = artifacts(model_setup, reported, splits["test"])
    return result, selection, model_setup.trained


def restore(path: Path, device: torch.device) -> Setup:
    """The Setup of a final run with the trained parameters of its checkpoint loaded."""
    saved = load_checkpoint(path)
    candidate = Candidate(**saved["run"]["config"]["candidate"])
    settings = TIERS[Tier(saved["run"]["tier"])]
    data = make_data(settings.data, saved["run"]["config"]["data_seed"])
    model_setup = setup(
        candidate, data, data.train.to(device), device, saved["run"]["seed"], settings
    )
    model_setup.trained.load_state_dict(saved["state_dict"])
    return model_setup


def artifacts(model_setup: Setup, predictions: Predictions, test: Batch) -> dict:
    """Test predictions on the first rows, and energy data of an energy model, for the plots."""
    rows = slice(0, ARTIFACT_ROWS)
    result = {
        "predictions": {
            "x": test.context[rows].squeeze(-1).cpu().tolist(),
            "y": test.target[rows].squeeze(-1).cpu().tolist(),
            "final": predictions.final[rows].squeeze(-1).tolist(),
            "initial": None
            if predictions.initial is None
            else predictions.initial[rows].squeeze(-1).tolist(),
        }
    }
    energy = model_setup.energy
    if energy is not None:
        result["energy_grid"] = {
            "x": GRID_X.tolist(),
            "y": GRID_Y.tolist(),
            "energy": energy_grid(energy, GRID_X, GRID_Y).tolist(),
        }
        x, y = test.context, test.target
        result["energy_gradient"] = {
            "on_branches": energy_gradient_magnitude(energy, x, y),
            "at_zero": energy_gradient_magnitude(energy, x, torch.zeros_like(y)),
            "step_size": model_setup.step_size,
        }
    for name, build in model_setup.extra_artifacts.items():
        result[name] = build()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--stage", choices=["search", "final", "all"], default="all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--reuse-stale", action="store_true", help=runner.REUSE_HELP)
    args = parser.parse_args()
    tier, device = Tier(args.tier), select_device(args.device)

    def train_one(candidate: Candidate, seed: int, with_test: bool):
        return run(candidate, tier, seed, device, with_test)

    runner.execute(
        args.output or Path("results/regression/g2") / tier.value,
        args.stage,
        tuple(args.models),
        lambda model: search_space(model, tier),
        Candidate,
        train_one,
        SEARCH_SEED,
        REPORT_SEEDS,
        runner.Provenance(code_fingerprint("regression_g2"), args.reuse_stale),
        checkpoint_directory(EXPERIMENT, tier.value),
    )


if __name__ == "__main__":
    main()
