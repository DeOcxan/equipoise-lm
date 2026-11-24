"""Question G5 on the saved final checkpoints: G3 out of distribution (method.md 20).

    python -u -m ebt.experiments.inference_g5 --task regression --tier full --device cpu
    python -u -m ebt.experiments.inference_g5 --task mnist --tier full --device mps
    python -u -m ebt.experiments.inference_g5 --task text --tier full --device mps

Loads the final checkpoints of the EBTs of one task and never trains. For every checkpoint it
measures the G3 curve of method.md 19.2 (the error after 1 to 64 inference steps from the same
initial candidates, with no noise) on out-of-distribution data:

- regression: the out-of-distribution split of method.md 16.1 (x in [4, 9));
- MNIST: the test images noised at the levels of MNIST_LEVELS (method.md 20.3);
- text: the out-of-distribution block of method.md 18.1.

No G4 value is computed. The in-distribution side of G5 is the existing G3 results of
`inference_g3_g4`; tools/g5_report.py compares the two.

The split a curve is measured on is chosen by name through `split_of`, which returns the rows of
that split of a task's data as an `EvaluationSplit`; no data object is changed. Regression uses
the G3 function of `inference_regression` on those rows. Text uses `text_g3`, the G3 part of
`inference_text.evaluate` as a function of the rows (`evaluate` there always computes G4 as
well). MNIST uses `mnist_g3` on images noised by `noisy_images` at a given level.

This module imports the three task modules of method.md 19, so its code fingerprint changes
with any of them and with this module; the fingerprints of the G3 and G4 results do not change
with this module. Results: one file per checkpoint, written as soon as it is finished, in
results/<task>/g5/<tier>/ (MNIST: one subdirectory noise-<level> per level), with summary.json.
A rerun skips a current result and stops at a stale one unless --reuse-stale, as in
`inference_g3_g4`.
"""

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ebt.checkpoints import checkpoint_directory, checkpoint_path, load_checkpoint
from ebt.config import Tier
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import (
    inference_mnist,
    inference_regression,
    inference_text,
    mnist_denoising,
    regression_g2,
    runner,
    text_lm,
)
from ebt.fingerprint import code_fingerprint
from ebt.results import environment, read_json, write_json
from ebt.tasks.mnist import add_noise, alpha_bar
from ebt.tasks.regression import make_data as make_regression_data
from ebt.training.batch import Batch

EXPERIMENT = "inference_g5"
SPLITS = ("test", "ood")
"""Names of the splits `split_of` can return."""
MNIST_LEVELS = {0.15: 40_002, 0.2: 40_003, 0.3: 40_004}
"""Out-of-distribution noise levels of MNIST and the seed of the fixed noise of each
(method.md 20.3); training and the in-distribution results use level 0.1 with seed 40 001."""
PRIMARY_LEVEL = 0.2
"""The MNIST level of the G5 decision (method.md 20.3)."""


# The split adapter ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSplit:
    """The rows a G3 curve is measured on and the name of the split they come from."""

    name: str
    rows: Batch

    def first(self, rows: int | None) -> "EvaluationSplit":
        """The first `rows` rows (all of them for None), under the same name."""
        if rows is None:
            return self
        return EvaluationSplit(self.name, self.rows.rows(slice(0, rows)))


def split_of(data, name: str) -> EvaluationSplit:
    """The split `name` ("test" or "ood") of a task's data (`RegressionData` or `TextData`),
    read from its attribute of that name. Raises if the name is unknown or the data have no
    such split."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    rows = getattr(data, name)
    if rows is None:
        raise ValueError(f"the data have no {name} split")
    return EvaluationSplit(name, rows)


# G3 curves of one checkpoint --------------------------------------------------------------------


def regression_curve(path: Path, device: torch.device, split_name: str, rows: int | None) -> dict:
    """G3 of one regression EBT checkpoint on the split `split_name` of its own data, with the
    G3 function of `inference_regression` (initial candidates N(0, s²), s of the training split,
    from the g3_initial seed)."""
    saved = load_checkpoint(path)
    seed = saved["run"]["seed"]
    settings = regression_g2.TIERS[Tier(saved["run"]["tier"])]
    data = make_regression_data(settings.data, saved["run"]["config"]["data_seed"])
    split = split_of(data, split_name).first(rows)
    setup = regression_g2.restore(path, device)
    x = split.rows.context
    return {
        "error": "branch_distance",
        "split": {"name": split.name, "rows": len(split.rows)},
        "x_range": [x.min().item(), x.max().item()],
        "g3": inference_regression.regression_g3(setup, split.rows, data.init_std, seed, device),
        "trained_thinking": saved["run"]["config"]["thinking"],
        "trained_inference": saved["run"]["config"]["inference"],
    }


def text_g3(model, loop, rows: Batch, batch: int, seed: int, device: torch.device) -> dict:
    """BPC over every position of `rows` per step count of STEP_GRID, from N(0, 1) initial
    logits of the g3_initial seed: the G3 part of `inference_text.evaluate`, for any rows."""
    vocabulary = model.config.vocabulary
    generator = torch.Generator().manual_seed(core.evaluation_seeds(seed)["g3_initial"])
    bits = {k: 0.0 for k in core.STEP_GRID}
    undefined = None
    for start in range(0, len(rows), batch):
        part = rows.rows(slice(start, start + batch))
        context, target = part.context.to(device), part.target

        def record(k, candidate, target=target):
            bits[k] += inference_text.position_bits(candidate.cpu(), target).sum().item()

        count_rows, length = context.shape
        initial = text_lm.initial_logits(count_rows, length, vocabulary, generator).to(device)
        stop = core.think(model, loop, context, initial, core.STEP_GRID, record)
        if stop is not None:
            undefined = stop if undefined is None else min(undefined, stop)
    positions = rows.target.numel()
    bpc = {k: bits[k] / positions for k in core.STEP_GRID}
    return {
        "steps": list(core.STEP_GRID),
        "bpc": core.curve(bpc, undefined),
        "undefined_from_step": undefined,
        "positions": positions,
    }


def text_curve(path: Path, device: torch.device, data, split_name: str, rows: int | None) -> dict:
    """G3 of one text EBT checkpoint on the split `split_name` of `data`."""
    saved = load_checkpoint(path)
    settings = text_lm.TIERS[Tier(saved["run"]["tier"])]
    model, loop = text_lm.restore(path, device)
    split = split_of(data, split_name).first(rows)
    g3 = text_g3(model, loop, split.rows, settings.eval_batch, saved["run"]["seed"], device)
    return {
        "error": "bits per character of each position",
        "split": {"name": split.name, "rows": len(split.rows)},
        "g3": g3,
        "trained_thinking": saved["run"]["config"]["thinking"],
        "trained_inference": saved["run"]["config"]["inference"],
    }


def noisy_images(clean: Tensor, level: float, seed: int) -> Tensor:
    """The images noised once at noise level `level` (method.md 17.2) from a CPU generator with
    `seed`: √ᾱ x + √(1 − ᾱ) ε with ᾱ = alpha_bar(level)."""
    return add_noise(clean, alpha_bar(level), torch.Generator().manual_seed(seed))


def mnist_g3(model, loop, clean: Tensor, noisy: Tensor, batch: int, device: torch.device) -> dict:
    """PSNR per step count of STEP_GRID, thinking from the noisy images as trained; one MSE
    over every pixel per step count (the G3 of `inference_mnist.evaluate`, for any images)."""
    squared = {k: 0.0 for k in core.STEP_GRID}
    undefined = None
    for start in range(0, len(clean), batch):
        part = slice(start, start + batch)

        def record(k, candidate, part=part):
            squared[k] += inference_mnist.pixel_mse(candidate, clean[part]).sum().item()

        stop = core.think(model, loop, None, noisy[part].to(device), core.STEP_GRID, record)
        if stop is not None:
            undefined = stop if undefined is None else min(undefined, stop)
    psnr = {k: 10 * math.log10(255**2 / (squared[k] / len(clean))) for k in core.STEP_GRID}
    return {
        "steps": list(core.STEP_GRID),
        "psnr": core.curve(psnr, undefined),
        "undefined_from_step": undefined,
    }


def mnist_curve(
    path: Path, device: torch.device, data, level: float, noise_seed: int, rows: int | None
) -> dict:
    """G3 of one MNIST EBT checkpoint on the test images noised at `level` with `noise_seed`."""
    saved = load_checkpoint(path)
    settings = mnist_denoising.TIERS[Tier(saved["run"]["tier"])]
    model, loop = mnist_denoising.restore(path, device)
    clean = data.test.target if rows is None else data.test.target[:rows]
    noisy = noisy_images(data.test.target, level, noise_seed)[: len(clean)]
    return {
        "error": "pixel MSE over every pixel (PSNR)",
        "split": {
            "name": "test",
            "rows": len(clean),
            "noise_level": level,
            "noise_seed": noise_seed,
        },
        "g3": mnist_g3(model, loop, clean, noisy, settings.eval_batch, device),
        "trained_thinking": saved["run"]["config"]["thinking"],
        "trained_inference": saved["run"]["config"]["inference"],
    }


# Tasks and files --------------------------------------------------------------------------------


TASKS = {
    "regression": core.Task(
        name="regression", module=EXPERIMENT, experiment=regression_g2.EXPERIMENT,
        models=tuple(inference_regression.G3_MODELS),
        report_seeds=tuple(regression_g2.REPORT_SEEDS), metric="branch_distance",
        lower_is_better=True, results="results/regression",
        evaluate=None, load_data=lambda tier, directory: None,
    ),
    "mnist": core.Task(
        name="mnist", module=EXPERIMENT, experiment=mnist_denoising.EXPERIMENT, models=("ebt",),
        report_seeds=tuple(mnist_denoising.REPORT_SEEDS), metric="psnr", lower_is_better=False,
        results="results/mnist", evaluate=None, load_data=inference_mnist.load,
    ),
    "text": core.Task(
        name="text", module=EXPERIMENT, experiment=text_lm.EXPERIMENT, models=("ebt",),
        report_seeds=tuple(text_lm.REPORT_SEEDS), metric="bpc", lower_is_better=True,
        results="results/text", evaluate=None, load_data=inference_text.load,
    ),
}  # fmt: skip
"""The task settings of G5 for `inference_g3_g4.summarize`; `module` is this module, whose
fingerprint the results carry. `evaluate` is not used: `execute` is given the evaluation."""


def execute(
    task: core.Task,
    evaluate,
    split: dict,
    models: Sequence[str],
    seeds: Sequence[int],
    checkpoints: Path,
    output: Path,
    device: torch.device,
    reuse_stale: bool,
    rows: int | None = None,
) -> dict:
    """Evaluate `evaluate(path)` for every checkpoint of `models` × `seeds` and write the
    summary. `split` describes the data; it is recorded as `settings` and is part of what a
    current result must match, with the code, the checkpoint and the row limit
    (`inference_g3_g4.provenance`). Stale results
    and checkpoints from other code stop the run before any work, unless `reuse_stale`."""
    fingerprint = code_fingerprint(EXPERIMENT)
    training_fingerprint = code_fingerprint(task.experiment)
    jobs = [
        core.Job(model, seed, checkpoint_path(checkpoints, model, seed), output)
        for model in models
        for seed in seeds
    ]
    missing = [job.checkpoint for job in jobs if not job.checkpoint.exists()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"{len(missing)} final checkpoints are missing:\n{listing}")
    wanted = {job: {**core.provenance(job, fingerprint, rows), "settings": split} for job in jobs}
    old_checkpoints = [
        f"{job.checkpoint} (code {wanted[job]['checkpoint']['code_fingerprint']})"
        for job in jobs
        if wanted[job]["checkpoint"]["code_fingerprint"] != training_fingerprint
    ]
    stale = [
        job
        for job in jobs
        if job.path.exists() and not core.current(read_json(job.path), wanted[job])
    ]
    found = old_checkpoints + [str(job.path) for job in stale]
    if found and not reuse_stale:
        runner.raise_stale(found)
    for job in jobs:
        if job.path.exists() and (job in stale or core.current(read_json(job.path), wanted[job])):
            continue
        began = time.perf_counter()
        result = evaluate(job.checkpoint)
        payload = {
            "experiment": EXPERIMENT,
            "question": "G5",
            "task": task.name,
            "fingerprint_module": EXPERIMENT,
            "model": job.model,
            "seed": job.seed,
            "device": device.type,
            "environment": environment(),
            "evaluation_seeds": core.evaluation_seeds(job.seed),
            "checkpoint_from_other_code": wanted[job]["checkpoint"]["code_fingerprint"]
            != training_fingerprint,
            "seconds": time.perf_counter() - began,
            **wanted[job],
            **result,
        }
        write_json(job.path, payload)
        runner.status(f"G5 {task.name} {job.path.relative_to(output.parent)} "
                      f"({payload['seconds']:.1f} s)")  # fmt: skip
    summary = core.summarize(task, models, seeds, output)
    summary["settings"] = split
    summary["reused_stale_files"] = sorted(str(job.path) for job in stale)
    summary["code_fingerprint"] = fingerprint
    summary["fingerprint_module"] = EXPERIMENT
    write_json(output / "summary.json", summary)
    return summary


def level_directory(output: Path, level: float) -> Path:
    return output / f"noise-{level:g}"


def run_task(
    name: str,
    tier: Tier,
    device: torch.device,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    data_directory: Path | None = None,
    checkpoints: Path | None = None,
    output: Path | None = None,
    reuse_stale: bool = False,
    rows: int | None = None,
    levels: Sequence[float] | None = None,
) -> dict:
    """Run G5 for one task; returns the summary of each output directory by its path."""
    task = TASKS[name]
    models = list(models or task.models)
    seeds = list(seeds or task.report_seeds)
    checkpoints = checkpoints or checkpoint_directory(task.experiment, tier.value)
    output = output or Path(task.results) / "g5" / tier.value
    if name == "mnist":
        data = task.load_data(tier, data_directory)
        jobs = []
        for level in levels or sorted(MNIST_LEVELS):
            seed = MNIST_LEVELS[level]
            split = {"name": "test", "noise_level": level, "noise_seed": seed}

            def evaluate(path, level=level, seed=seed):
                return mnist_curve(path, device, data, level, seed, rows)

            jobs.append((evaluate, split, level_directory(output, level)))
    elif name == "text":
        data = task.load_data(tier, data_directory)

        def evaluate(path):
            return text_curve(path, device, data, "ood", rows)

        jobs = [(evaluate, {"name": "ood"}, output)]
    else:

        def evaluate(path):
            return regression_curve(path, device, "ood", rows)

        jobs = [(evaluate, {"name": "ood"}, output)]
    return {
        str(directory): execute(
            task, evaluate, split, models, seeds, checkpoints, directory, device, reuse_stale, rows
        )
        for evaluate, split, directory in jobs
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--levels", nargs="+", type=float, default=None,
                        help=f"MNIST noise levels, a subset of {sorted(MNIST_LEVELS)}")  # fmt: skip
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None, help="evaluate the first rows only")
    parser.add_argument("--reuse-stale", action="store_true", help=runner.REUSE_HELP)
    args = parser.parse_args()
    unknown = sorted(set(args.levels or ()) - set(MNIST_LEVELS))
    if unknown:
        parser.error(f"unknown noise levels {unknown}; expected a subset of {sorted(MNIST_LEVELS)}")
    run_task(args.task, Tier(args.tier), torch.device(args.device), args.models, args.seeds,
             args.data, args.checkpoints, args.output, args.reuse_stale, args.rows,
             args.levels)  # fmt: skip


if __name__ == "__main__":
    main()
