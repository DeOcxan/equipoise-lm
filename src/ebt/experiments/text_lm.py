"""Character-level language model: a causal S2 EBT against a causal transformer (method.md 18).

    python -u -m ebt.experiments.text_lm --tier full --device mps --stage search

Hyperparameters are chosen on validation data by each model's own objective, the validation
bits per character at its best checkpoint (method.md section 18.5); the test split and the
out-of-distribution text are used only by the final runs. Every search configuration and
every final run is written to disk as soon as it finishes, and a rerun skips the work already
on disk. Final runs save their best checkpoint.
"""

import argparse
import copy
import dataclasses
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn

from ebt.checkpoints import checkpoint_directory, load_checkpoint
from ebt.config import (
    InferenceConfig,
    ObjectiveConfig,
    ReplayConfig,
    ThinkingConfig,
    Tier,
    TrainingConfig,
    TrainingMode,
)
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import runner
from ebt.fingerprint import code_fingerprint
from ebt.models.mlp import parameter_count
from ebt.models.text_transformer import CausalTransformer, TextEnergy, TextTransformerConfig
from ebt.results import RunResult, environment, read_json
from ebt.runtime import make_generator, run_seed, seed_everything, select_device
from ebt.tasks.text import (
    CORPUS_SHA256,
    OOD_SHA256,
    TextData,
    bits_per_character,
    cross_entropy,
    load_data,
)
from ebt.training.batch import Batch
from ebt.training.evaluation import Predictions, predict_ebt, predict_feedforward
from ebt.training.objectives import EBTObjective, FeedForwardObjective
from ebt.training.trainer import train

EXPERIMENT = "text_lm"
MODELS = ("ebt", "transformer")
SEARCH_SEED = 100
REPORT_SEEDS = (0, 1, 2, 3, 4)
FIXED_SEEDS = {"validation_initial": 50_000, "test_initial": 50_001, "ood_initial": 50_002}
"""Seeds of the initial logits of the EBT on each evaluation split, the same for every run, so
that every configuration and checkpoint is evaluated from the same starting point."""
RUN_SOURCES = ("initialization", "batch", "noise", "replay", "generation")
"""The random sources of one run, each with its own seed (`run_seeds`)."""


def run_seeds(seed: int) -> dict[str, int]:
    """Seed of each random source of run `seed`: `run_seed` of the source's index, disjoint
    across sources and runs and from FIXED_SEEDS."""
    return {name: run_seed(index, seed) for index, name in enumerate(RUN_SOURCES)}


INFERENCE = InferenceConfig(steps=4)
"""Four steps, the maximum of the training range {2, 3, 4}, without noise (method.md 18.4)."""
STEP_SIZE_JITTER = 2.0
STEPS_MIN, STEPS_MAX = 2, 4
LANGEVIN_STD = 3.0
REPLAY = ReplayConfig()
WARMUP_FRACTION = 0.1
PROMPT_LENGTH = 32
"""Generation prompt: the first characters of the validation split."""
SAMPLES, SAMPLE_LENGTH = 2, 500


@dataclass(frozen=True)
class TierSettings:
    dim: int
    depth: int
    layout: str
    sequence_length: int
    batch_size: int
    train_steps: int
    validate_every: int
    eval_batch: int
    transformer_learning_rates: tuple[float, ...]
    step_sizes: tuple[float, ...]
    """α grid of the first EBT search stage, at `ebt_learning_rate`."""
    ebt_learning_rate: float
    second_stage_learning_rate: float
    """Learning rate of the second EBT stage, at the best α of the first (method.md 18.5)."""
    sample_length: int


TIERS = {
    Tier.FAST: TierSettings(
        dim=64,
        depth=1,
        layout="split",
        sequence_length=32,
        batch_size=8,
        train_steps=20,
        validate_every=10,
        eval_batch=64,
        transformer_learning_rates=(1e-3,),
        step_sizes=(0.5, 5.0),
        ebt_learning_rate=1.2e-3,
        second_stage_learning_rate=3e-3,
        sample_length=16,
    ),
    Tier.FULL: TierSettings(
        dim=256,
        depth=6,
        layout="split",
        sequence_length=128,
        batch_size=32,
        train_steps=2_000,
        validate_every=200,
        eval_batch=32,
        transformer_learning_rates=(3e-4, 1e-3, 3e-3, 1e-2),
        step_sizes=(0.5, 5.0, 50.0),
        ebt_learning_rate=1.2e-3,
        second_stage_learning_rate=3e-3,
        sample_length=SAMPLE_LENGTH,
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One hyperparameter configuration; `step_size` is the EBT's fixed α."""

    model: str
    learning_rate: float
    step_size: float | None = None


def search_space(model: str, tier: Tier, output: Path) -> list[Candidate]:
    """The configurations tried on validation data, in grid order (method.md section 18.5).

    The EBT search has two stages. The first tries every α of the grid at the authors'
    learning rate. Once all of them are on disk in `output`, the second stage adds one
    configuration: the second-stage learning rate at the α chosen among the first stage by
    `runner.choose`. The selection is then made over all EBT configurations.
    """
    settings = TIERS[tier]
    if model == "transformer":
        return [Candidate(model, lr) for lr in settings.transformer_learning_rates]
    first = [Candidate(model, settings.ebt_learning_rate, alpha) for alpha in settings.step_sizes]
    paths = [runner.search_path(output, model, candidate) for candidate in first]
    if not all(path.exists() for path in paths):
        return first
    alpha = runner.choose([read_json(path) for path in paths])["candidate"]["step_size"]
    return [*first, Candidate(model, settings.second_stage_learning_rate, alpha)]


def model_config(settings: TierSettings, vocabulary: int) -> TextTransformerConfig:
    return TextTransformerConfig(vocabulary, settings.dim, settings.depth, settings.layout)


def thinking_config(candidate: Candidate) -> ThinkingConfig:
    """S2 with fixed α, energy-scaled step, jitter 2, k ∈ {2, 3, 4} and Langevin σ = 3."""
    return ThinkingConfig(
        mode=TrainingMode.S2,
        step_size=candidate.step_size,
        step_size_jitter=STEP_SIZE_JITTER,
        energy_scaled_step=True,
        steps_min=STEPS_MIN,
        steps_max=STEPS_MAX,
        langevin_std=LANGEVIN_STD,
    )


def training_config(candidate: Candidate, settings: TierSettings) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=candidate.learning_rate,
        train_steps=settings.train_steps,
        batch_size=settings.batch_size,
        warmup_steps=int(WARMUP_FRACTION * settings.train_steps),
    )


def initial_logits(rows: int, length: int, vocabulary: int, generator: torch.Generator) -> Tensor:
    """Candidate logits drawn from N(0, 1) (method.md section 4.1), on the generator's device."""
    return torch.randn((rows, length, vocabulary), generator=generator, device=generator.device)


def predict(
    model: nn.Module, loop: ThinkingLoop | None, split: Batch, seed: int, settings, device
) -> Predictions:
    """Next-character logits for every window of `split`; for the EBT after the inference
    steps from the initial logits of a CPU generator seeded with `seed`."""
    split = split.to(device)
    if loop is None:
        return predict_feedforward(model, split.context, settings.eval_batch)
    rows, length = split.context.shape
    vocabulary = model.config.vocabulary
    initial = initial_logits(rows, length, vocabulary, torch.Generator().manual_seed(seed))
    return predict_ebt(
        model, loop, split, initial.to(device), INFERENCE, None, settings.eval_batch, False
    )


def run(
    candidate: Candidate,
    tier: Tier,
    seed: int,
    device: torch.device,
    with_test: bool,
    data: TextData,
) -> tuple[RunResult, tuple[str, float], nn.Module]:
    """Train one model with one seed; evaluate its best checkpoint on validation data and, if
    requested, on test and out-of-distribution data."""
    settings = TIERS[tier]
    seeds = run_seeds(seed)
    seed_everything(seeds["initialization"])
    batch_generator = torch.Generator().manual_seed(seeds["batch"])
    noise_generator = make_generator(seeds["noise"], device)
    training = training_config(candidate, settings)
    config = model_config(settings, len(data.vocabulary))

    loop = None
    thinking = None
    if candidate.model == "ebt":
        model = TextEnergy(config).to(device)
        thinking = thinking_config(candidate)
        loop = ThinkingLoop(thinking).to(device)
        replay = ReplayBuffer(REPLAY, torch.Generator().manual_seed(seeds["replay"]))

        def initial(batch: Batch, generator: torch.Generator) -> Tensor:
            rows, length = batch.context.shape
            return initial_logits(rows, length, config.vocabulary, generator)

        objective = EBTObjective(model, loop, cross_entropy, ObjectiveConfig(), initial, replay)
    else:
        model = CausalTransformer(config).to(device)
        objective = FeedForwardObjective(model, cross_entropy)

    def validation_bpc() -> float:
        predictions = predict(
            model, loop, data.validation, FIXED_SEEDS["validation_initial"], settings, device
        )
        return bits_per_character(predictions.final, data.validation.target)

    best = {"bpc": float("inf"), "step": None, "state": None, "evaluations": 0}

    def validate() -> dict[str, float]:
        """Validation BPC; the parameters with the lowest value so far are kept in memory."""
        value = validation_bpc()
        best["evaluations"] += 1
        if value < best["bpc"]:
            best.update(bpc=value, step=best["evaluations"] * settings.validate_every)
            best["state"] = copy.deepcopy(objective.module.state_dict())
        return {"bpc": value}

    history = train(
        objective,
        data.train.to(device),
        training,
        batch_generator,
        noise_generator,
        validate=validate,
        validate_every=settings.validate_every,
    )
    if best["state"] is None:
        raise ValueError("no validation ran: train_steps is below validate_every")
    objective.module.load_state_dict(best["state"])

    splits = {"validation": (data.validation, FIXED_SEEDS["validation_initial"])}
    if with_test:
        splits["test"] = (data.test, FIXED_SEEDS["test_initial"])
        if data.ood is not None:
            splits["ood"] = (data.ood, FIXED_SEEDS["ood_initial"])
    predictions = {
        name: predict(model, loop, split, initial_seed, settings, device)
        for name, (split, initial_seed) in splits.items()
    }
    metrics = {
        name: {"bpc": bits_per_character(predictions[name].final, split.target)}
        for name, (split, _) in splits.items()
    }
    selection = ("validation_bpc_best_checkpoint", metrics["validation"]["bpc"])
    reported = predictions["test" if with_test else "validation"]
    ood = data.ood_text
    result = RunResult(
        experiment=EXPERIMENT,
        task="text",
        model=candidate.model,
        tier=tier.value,
        seed=seed,
        device=device.type,
        dtype="float32",
        environment=environment(),
        config={
            "candidate": dataclasses.asdict(candidate),
            "model": dataclasses.asdict(config),
            "sequence_length": settings.sequence_length,
            "training": dataclasses.asdict(training),
            "thinking": dataclasses.asdict(thinking) if thinking else None,
            "inference": dataclasses.asdict(INFERENCE) if loop else None,
            "replay": dataclasses.asdict(REPLAY) if loop else None,
            "run_seeds": seeds,
            "fixed_seeds": FIXED_SEEDS,
            "corpus_sha256": CORPUS_SHA256,
            "ood_sha256": None if ood is None else OOD_SHA256,
            "ood_text": None
            if ood is None
            else {
                "body_characters": ood.body_characters,
                "novel_characters": ood.novel_characters,
                "dropped": ood.dropped,
                "dropped_fraction": ood.dropped_fraction,
                "block_characters": len(ood.block),
            },
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
            "validation_bpc": history.validation.get("bpc", []),
            "best_step": best["step"],
        },
    )
    if with_test and seed == REPORT_SEEDS[0]:
        prompt = data.validation.context[0, :PROMPT_LENGTH]
        generator = torch.Generator().manual_seed(seeds["generation"])
        samples = generate(model, loop, prompt, settings, generator, device)
        result.artifacts = {
            "samples": {
                "prompt": data.vocabulary.decode(prompt),
                "texts": [data.vocabulary.decode(sample) for sample in samples],
            }
        }
    return result, selection, objective.module


def generate(
    model: nn.Module,
    loop: ThinkingLoop | None,
    prompt: Tensor,
    settings: TierSettings,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """SAMPLES continuations of `prompt`, sampled one character at a time at temperature 1.

    The window is the last `sequence_length` characters. The EBT thinks INFERENCE.steps steps
    from N(0, 1) logits for the whole window and samples from its final logits at the last
    position. Illustration only, not a metric. Returns the generated indices (SAMPLES, n).
    """
    tokens = prompt.to(device).repeat(SAMPLES, 1)
    vocabulary = model.config.vocabulary
    for _ in range(settings.sample_length):
        window = tokens[:, -settings.sequence_length :]
        if loop is None:
            with torch.no_grad():
                logits = model(window)[:, -1]
        else:
            initial = initial_logits(SAMPLES, window.shape[1], vocabulary, generator)
            trajectory = loop.infer(model, window, initial.to(device), INFERENCE)
            logits = trajectory.candidates[-1][:, -1]
        probabilities = logits.cpu().double().softmax(dim=-1)
        chosen = torch.multinomial(probabilities, 1, generator=generator)
        tokens = torch.cat((tokens, chosen.to(device=device, dtype=tokens.dtype)), dim=1)
    return tokens[:, len(prompt) :].cpu()


def restore(path: Path, device: torch.device) -> tuple[nn.Module, ThinkingLoop | None]:
    """The best checkpoint of a final run: the model and, for the EBT, its thinking loop."""
    saved = load_checkpoint(path)
    config = saved["run"]["config"]
    model_settings = TextTransformerConfig(**config["model"])
    candidate = Candidate(**config["candidate"])
    if candidate.model == "transformer":
        model = CausalTransformer(model_settings).to(device)
        model.load_state_dict(saved["state_dict"])
        return model, None
    model = TextEnergy(model_settings).to(device)
    loop = ThinkingLoop(thinking_config(candidate)).to(device)
    nn.ModuleList([model, loop]).load_state_dict(saved["state_dict"])
    return model, loop


def execute(
    output: Path,
    checkpoints: Path,
    stage: str,
    models,
    tier: Tier,
    device,
    data,
    reuse_stale: bool,
):
    """The search in two calls of `runner.execute` (the second adds the second EBT stage once
    the first stage is on disk), then the final runs, with checkpoints in `checkpoints`."""
    common = (
        tuple(models),
        lambda model: search_space(model, tier, output),
        Candidate,
        partial(_train_one, tier=tier, device=device, data=data),
        SEARCH_SEED,
        REPORT_SEEDS,
        runner.Provenance(code_fingerprint(EXPERIMENT), reuse_stale),
        checkpoints,
    )
    if stage in ("search", "all"):
        for _ in range(2):
            runner.execute(output, "search", *common)
    if stage in ("final", "all"):
        runner.execute(output, "final", *common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--stage", choices=["search", "final", "all"], default="all")
    parser.add_argument("--data", type=Path, default=Path("data/text/raw"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--reuse-stale", action="store_true", help=runner.REUSE_HELP)
    args = parser.parse_args()
    tier, device = Tier(args.tier), select_device(args.device)
    data = load_data(args.data, TIERS[tier].sequence_length)
    output = args.output or Path("results/text/lm") / tier.value
    checkpoints = checkpoint_directory(EXPERIMENT, tier.value)
    execute(output, checkpoints, args.stage, args.models, tier, device, data, args.reuse_stale)


def _train_one(candidate, seed, with_test, *, tier, device, data):
    return run(candidate, tier, seed, device, with_test, data)


if __name__ == "__main__":
    main()
