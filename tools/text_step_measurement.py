"""Training step time and peak memory of one text setting (method.md section 18.8).

usage: python -m tools.text_step_measurement --dim DIM --depth DEPTH --seq SEQ --batch BATCH
       --layout {split,full} --model {ebt,transformer} [--device mps] [--vocabulary 65]
       [--validation-characters 55770] [--warmup 2] [--repeats 5]

Measures one setting with the modules of this repository, on random tokens (`measure`). Run one
setting per process, so that the driver memory belongs to that setting only; `tools.text_budget`
does this.

Output: one JSON record on standard output. An MPS out-of-memory error ends the process with a
RuntimeError whose message contains "out of memory".
"""

import argparse
import json
import resource
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

from ebt.config import ObjectiveConfig, ThinkingConfig, TrainingConfig, TrainingMode
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import text_lm
from ebt.models.text_transformer import CausalTransformer, TextEnergy, TextTransformerConfig
from ebt.tasks.text import cross_entropy
from ebt.training.batch import Batch
from ebt.training.evaluation import predict_ebt, predict_feedforward
from ebt.training.objectives import EBTObjective, FeedForwardObjective, TrainingObjective
from ebt.training.optimizer import build_optimizer

GIB = 2**30
WARMUP, REPEATS = 2, 5
"""Default training steps per measured step count: warm-up steps, then timed steps."""
STEP_COUNTS = (2, 3, 4)
"""The EBT's fixed step counts k, each measured with WARMUP + REPEATS training steps."""
GENERATIONS = 10
"""Timed EBT inferences for the generation time per character."""


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_gib(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / GIB
    # ru_maxrss is in bytes on macOS and in kibibytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / GIB


def memory_source(device: torch.device) -> str:
    return "mps driver allocation" if device.type == "mps" else "process peak resident set size"


def timed(step: Callable[[], None], warmup: int, repeats: int, device: torch.device) -> float:
    """Mean seconds of one call of `step` over `repeats` calls, after `warmup` calls."""
    for _ in range(warmup):
        step()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def trainer_step(
    objective: TrainingObjective, batch: Batch, generator: torch.Generator, config: TrainingConfig
) -> Callable[[], None]:
    """One training step: loss, backward, gradient clip at 1 and an AdamW step."""
    optimizer = build_optimizer(objective.module, objective.loop, config)
    parameters = [p for p in objective.module.parameters() if p.requires_grad]

    def step() -> None:
        result = objective.batch_loss(batch, generator)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return step


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--model", choices=["ebt", "transformer"], required=True)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--vocabulary", type=int, default=65)
    parser.add_argument("--validation-characters", type=int, default=55_770)
    return parser.parse_args(argv)


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """The record of one setting.

    - Training step (loss, backward, gradient clip, AdamW): mean of `repeats` steps after
      `warmup` steps. For the EBT the step count k is fixed to 2, 3 and 4 in turn, with replay and
      the training noise of method.md section 18.4; `step_mean_s` is the mean over the three.
    - Memory: on MPS the driver allocation after the timed steps (the allocator cache is emptied
      before each k and kept during it, so the value includes the peak). On CPU, where there is
      no driver allocation, the peak resident set size of the process; `memory_source` says which.
    - Validation evaluation: one pass over `validation_characters` characters in windows of S,
      with the evaluation batch of the full tier.
    - Generation (EBT): one inference of 4 steps on a window of 2 rows of length S, per character.
    """
    device = torch.device(args.device)
    vocabulary = args.vocabulary
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(1)
    config = TextTransformerConfig(vocabulary, args.dim, args.depth, args.layout)
    tokens = torch.randint(0, vocabulary, (args.batch, args.seq + 1), dtype=torch.uint8).to(device)
    batch = Batch(tokens[:, :-1], tokens[:, 1:])
    validation_rows = (args.validation_characters - 1) // args.seq
    validation = torch.randint(0, vocabulary, (validation_rows, args.seq + 1), dtype=torch.uint8)
    validation = Batch(validation[:, :-1], validation[:, 1:]).to(device)
    training = TrainingConfig(
        learning_rate=1e-3, train_steps=10, batch_size=args.batch, warmup_steps=0
    )
    keys = ("dim", "depth", "seq", "batch", "layout", "model", "warmup", "repeats")
    result = {key: getattr(args, key) for key in keys} | {"heads": config.heads}
    result["device"] = args.device
    result["vocabulary"] = vocabulary
    result["validation_characters"] = args.validation_characters
    result["memory_source"] = memory_source(device)
    eval_batch = text_lm.TIERS[text_lm.Tier.FULL].eval_batch
    empty_cache(device)

    if args.model == "transformer":
        model = CausalTransformer(config).to(device)
        objective = FeedForwardObjective(model, cross_entropy)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        step = trainer_step(objective, batch, generator, training)
        result["step_s"] = timed(step, args.warmup, args.repeats, device)
        result["peak_driver_gib"] = peak_memory_gib(device)
        model.eval()
        start = time.perf_counter()
        predict_feedforward(model, validation.context, eval_batch)
        synchronize(device)
        result["validation_eval_s"] = time.perf_counter() - start
        return result

    model = TextEnergy(config).to(device)
    result["parameters"] = sum(p.numel() for p in model.parameters())
    for k in STEP_COUNTS:
        empty_cache(device)
        loop = ThinkingLoop(
            ThinkingConfig(
                mode=TrainingMode.S2,
                step_size=5.0,
                step_size_jitter=2.0,
                energy_scaled_step=True,
                steps_min=k,
                steps_max=k,
                langevin_std=3.0,
            )
        ).to(device)
        replay = ReplayBuffer(text_lm.REPLAY, torch.Generator().manual_seed(2))

        def initial(b: Batch, g: torch.Generator) -> torch.Tensor:
            return torch.randn((len(b), args.seq, vocabulary), generator=g, device=device)

        objective = EBTObjective(model, loop, cross_entropy, ObjectiveConfig(), initial, replay)
        step = trainer_step(objective, batch, generator, training)
        result[f"step_k{k}_s"] = timed(step, args.warmup, args.repeats, device)
        result[f"peak_driver_k{k}_gib"] = peak_memory_gib(device)
    result["step_mean_s"] = sum(result[f"step_k{k}_s"] for k in STEP_COUNTS) / len(STEP_COUNTS)
    model.eval()
    initial_validation = torch.randn((validation_rows, args.seq, vocabulary), device=device)
    synchronize(device)
    start = time.perf_counter()
    predict_ebt(
        model, loop, validation, initial_validation, text_lm.INFERENCE, None, eval_batch, False
    )
    synchronize(device)
    result["validation_eval_s"] = time.perf_counter() - start
    window = validation.context[:2]
    start = time.perf_counter()
    for _ in range(GENERATIONS):
        candidate = torch.randn((2, args.seq, vocabulary), device=device)
        loop.infer(model, window, candidate, text_lm.INFERENCE)
    synchronize(device)
    result["generation_s_per_character"] = (time.perf_counter() - start) / GENERATIONS
    return result


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(measure(parse(argv))), flush=True)


if __name__ == "__main__":
    main()
