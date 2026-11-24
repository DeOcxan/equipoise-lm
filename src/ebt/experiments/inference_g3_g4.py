"""Questions G3 and G4 on the saved final checkpoints (questions.md G3 and G4, method.md 19).

    python -u -m ebt.experiments.inference_g3_g4 --task regression --tier full --device cpu
    python -u -m ebt.experiments.inference_g3_g4 --task mnist --tier full --device mps
    python -u -m ebt.experiments.inference_g3_g4 --task text --tier full --device mps

Loads the final checkpoints of one task and never trains. For every checkpoint it measures:

- G3: the task error after 1 to 64 inference steps (STEP_GRID) from the same initial
  candidates, with no noise; for the regression model (c3) also the inference experiment of
  G3 (smaller step sizes and the accept-only-lower step);
- G4: the Spearman correlation between final energy and error, best-of-N selection by lowest
  energy against random and oracle selection, and for regression the both-branch coverage.

This module holds what the three tasks share and imports no task code. The evaluation of each
task is in its own module, `inference_regression`, `inference_mnist` and `inference_text`, which
imports this one and the experiment of its task only. A result carries the code fingerprint of
its task module (method.md 19.1), so a change in one task's code leaves the results of the other
tasks current. The command line loads the task module by name.

Each checkpoint gives one result file, written as soon as it is finished, with that code
fingerprint and the fingerprint and SHA-256 of the checkpoint. A rerun skips a result whose
fingerprints and settings equal the current ones. A result written by other code or from
another checkpoint is stale: the run stops before any work and lists it, unless --reuse-stale
is given, in which case it is kept and listed in the summary. summary.json and table.md collect
the seeds of every model.
"""

import argparse
import hashlib
import importlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ebt.checkpoints import checkpoint_directory, checkpoint_path, load_checkpoint, recorded_path
from ebt.config import InferenceConfig, Tier
from ebt.core.energy import EnergyModel
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import runner
from ebt.fingerprint import code_fingerprint
from ebt.results import environment, read_json, write_json
from ebt.statistics import mean_interval

EXPERIMENT = "inference_g3_g4"
TASK_MODULES = {
    "regression": "inference_regression",
    "mnist": "inference_mnist",
    "text": "inference_text",
}
"""The module in `ebt.experiments` of each task; its fingerprint is that of the task's results."""
STEP_GRID = (1, 2, 4, 8, 16, 32, 64)
"""Inference step counts of G3: 1 to 64, doubling (the planned roughly logarithmic grid)."""
CANDIDATE_GRID = (1, 2, 4, 8, 16, 32)
"""N of best-of-N and of the both-branch coverage (G4: 1 to 32 in powers of two)."""
EVALUATION_SEED_OFFSET = 2_000_000
EVALUATION_SOURCES = ("g3_initial", "g4_candidates")
"""Random sources of the evaluation of one checkpoint (`evaluation_seeds`)."""
UNDEFINED_TOLERANCE = 1e-5
"""Best-of-N fraction: undefined when the oracle's gain over random selection is at most this
times |random error|. The errors come from float32 candidates (resolution about 1.2e-7 relative);
the factor leaves room for the rounding accumulated over the inference steps, so a gain this
small is not distinguished from an oracle equal to random selection."""
SINGLE_STEP = InferenceConfig(steps=1)


def evaluation_seeds(seed: int) -> dict[str, int]:
    """Seed of each evaluation source for the checkpoint of run `seed`:
    2 000 000 + 1000 · i + seed for source number i, above every training seed
    (1 000 000 + 1000 · i + seed with i < 10) and every fixed seed (below 1 000 000)."""
    if not 0 <= seed < 1000:
        raise ValueError(f"run seed {seed} outside [0, 1000)")
    return {
        name: EVALUATION_SEED_OFFSET + 1000 * index + seed
        for index, name in enumerate(EVALUATION_SOURCES)
    }


# Metrics ---------------------------------------------------------------------------------------


def _require_finite(name: str, *tensors: Tensor) -> None:
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} received non-finite values")


def ranks(values: Tensor) -> Tensor:
    """Ranks 1..n in float64; tied values share the mean of their ranks. Finite values only."""
    _require_finite("ranks", values)
    values = values.double().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    result = torch.empty_like(values)
    start = 0
    n = len(values)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = (start + end + 1) / 2  # mean of ranks start+1 .. end
        start = end
    return result


def spearman(first: Tensor, second: Tensor) -> float | None:
    """Spearman's rank correlation: the Pearson correlation of the ranks (ties averaged).
    None when either variable is constant."""
    a, b = ranks(first), ranks(second)
    if len(a) != len(b):
        raise ValueError("the two variables differ in length")
    a, b = a - a.mean(), b - b.mean()
    denominator = (a.square().sum() * b.square().sum()).sqrt()
    if denominator == 0:
        return None
    return float((a * b).sum() / denominator)


def selection_fraction(random: float, oracle: float, chosen: float) -> float | None:
    """(random − chosen) / (random − oracle): 0 for no gain over random selection, 1 for the
    oracle's gain. None when the oracle's gain is at most UNDEFINED_TOLERANCE · |random|."""
    gain = random - oracle
    if gain <= UNDEFINED_TOLERANCE * abs(random):
        return None
    return (random - chosen) / gain


def best_of_n(errors: Tensor, energies: Tensor, sizes: Sequence[int]) -> list[dict]:
    """Best-of-N over the first N candidates of each row (errors and energies (rows, M)).

    For each N: the mean over rows of the error of the lowest-energy candidate (the first in
    candidate order among equal energies), of the expected error of a uniformly random
    candidate (the row mean), and of the lowest error (the oracle), with `selection_fraction`.
    Finite values only (`g4_non_finite`).
    """
    _require_finite("best_of_n", errors, energies)
    errors, energies = errors.double(), energies.double()
    rows = []
    for n in sizes:
        e, energy = errors[:, :n], energies[:, :n]
        chosen = e.gather(1, energy.argmin(dim=1, keepdim=True)).mean().item()
        random, oracle = e.mean().item(), e.min(dim=1).values.mean().item()
        rows.append(
            {
                "n": n,
                "random": random,
                "oracle": oracle,
                "lowest_energy": chosen,
                "fraction": selection_fraction(random, oracle, chosen),
            }
        )
    return rows


def undefined_best_of_n(sizes: Sequence[int], extra: Sequence[str] = ()) -> list[dict]:
    """The best-of-N rows of a checkpoint whose G4 values are undefined."""
    keys = ("random", "oracle", "lowest_energy", "fraction", *extra)
    return [{"n": n, **dict.fromkeys(keys)} for n in sizes]


def g4_non_finite(*tensors: Tensor | None) -> int:
    """Number of non-finite values among the G4 energies, errors and candidates. When it is not
    zero, every G4 value of the checkpoint is undefined and the number is recorded, as a
    non-finite step makes the later G3 values undefined."""
    return sum(int((~torch.isfinite(t)).sum()) for t in tensors if t is not None)


# G3 ranges --------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def classify_ranges(
    steps: Sequence[int], per_seed: Sequence[Sequence[float | None]], lower_is_better: bool
) -> dict:
    """The falling, flat, rising and undefined ranges of a G3 curve (method.md 19.2).

    `per_seed` holds one value per step for every seed; error is the value itself if
    `lower_is_better`, else its negative (PSNR). For two step counts i < j the change is the
    difference of the means over seeds and the spread the larger of their two sample standard
    deviations. The pair is level when the change is zero or smaller in magnitude than the
    spread, otherwise falling or rising by the sign of the change in error.

    A flat range has every pair of its step counts level. A falling (rising) range has falling
    (rising) end points and no rising (falling) pair. Ranges are cut from left to right; at the
    start a, with F the end of the longest flat range from a and M the end of the longest
    falling or rising range from a:

    1. if there is no falling or rising range from a, or F > M: the flat range a–F;
    2. else the falling (rising) range from a to the smallest c, at least two intervals before
       M, from which the rest up to M is flat while the change from a to c is still falling
       (rising); to M if there is no such c.

    The next range starts where this one ends, so a flat stretch of two or more intervals at
    the end of a change becomes its own range, while a flat stretch before a change is part of
    the change. Step counts from the first one with an
    undefined value of any seed on form one undefined range from the last defined step count
    to the end.
    """
    columns = [list(column) for column in zip(*per_seed, strict=True)]
    defined = 0
    while defined < len(columns) and all(
        v is not None and math.isfinite(v) for v in columns[defined]
    ):
        defined += 1
    means = [_mean(c) for c in columns[:defined]]
    spreads = [_std(c) for c in columns[:defined]]

    def kind(i: int, j: int) -> int:
        change = means[j] - means[i]
        if change == 0 or abs(change) < max(spreads[i], spreads[j]):
            return 0
        error_change = change if lower_is_better else -change
        return -1 if error_change < 0 else 1

    def flat(a: int, b: int) -> bool:
        return all(kind(i, j) == 0 for i in range(a, b + 1) for j in range(i + 1, b + 1))

    def monotone(a: int, b: int, sign: int) -> bool:
        return kind(a, b) == sign and all(
            kind(i, j) != -sign for i in range(a, b + 1) for j in range(i + 1, b + 1)
        )

    names = {-1: "falling", 0: "flat", 1: "rising"}
    ranges = []
    a = 0
    while a < defined - 1:
        longest_flat = max((b for b in range(a + 1, defined) if flat(a, b)), default=a)
        longest = max(
            ((b, s) for b in range(a + 1, defined) for s in (-1, 1) if monotone(a, b, s)),
            default=(a, 0),
        )
        last, sign = longest
        if sign == 0 or longest_flat > last:
            end, sign = longest_flat, 0
        else:
            end = next(
                (c for c in range(a + 1, last - 1) if flat(c, last) and kind(a, c) == sign), last
            )
        ranges.append({"kind": names[sign], "from": steps[a], "to": steps[end]})
        a = end
    if defined < len(steps):
        start = steps[defined - 1] if defined > 0 else steps[0]
        ranges.append({"kind": "undefined", "from": start, "to": steps[-1]})
    summary = {}
    for name in ("falling", "flat", "rising", "undefined"):
        found = [(r["from"], r["to"]) for r in ranges if r["kind"] == name]
        summary[name] = found or "absent"
    return {"ranges": ranges, **summary}


# Thinking with a record at the grid steps -------------------------------------------------------

Step = Callable[[EnergyModel, ThinkingLoop, Tensor | None, Tensor], Tensor]


def plain_step(
    model: EnergyModel, loop: ThinkingLoop, context: Tensor | None, candidate: Tensor
) -> Tensor:
    """One inference step of `loop` without noise."""
    return loop.infer(model, context, candidate, SINGLE_STEP).candidates[-1]


def think(
    model: EnergyModel,
    loop: ThinkingLoop,
    context: Tensor | None,
    initial: Tensor,
    steps: Sequence[int],
    record: Callable[[int, Tensor], None],
    step: Step = plain_step,
) -> int | None:
    """Single inference steps from `initial` up to max(steps), with no noise; `record(k, ŷ_k)`
    at every k in `steps`. One step at a time gives the same candidates as one call with k
    steps, since inference has no step-size or step-count randomness. `step` makes one step
    (the regression (c3) experiment passes the accept-only-lower step).

    Returns the first step whose candidate or gradient is not finite (the later grid values
    are then undefined), or None.
    """
    candidate = initial
    wanted = set(steps)
    for k in range(1, max(steps) + 1):
        try:
            candidate = step(model, loop, context, candidate)
        except FloatingPointError:
            return k
        if not torch.isfinite(candidate).all():
            return k
        if k in wanted:
            record(k, candidate)
    return None


def curve(values: dict[int, float], undefined_from: int | None) -> list[float | None]:
    """Values at STEP_GRID; None from the first undefined step on, that step included."""
    return [
        None if undefined_from is not None and k >= undefined_from else values[k] for k in STEP_GRID
    ]


# Tasks, files, provenance and the summary -------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """What the shared code needs from a task module.

    `evaluate(path, device, data, rows)` returns the result of one checkpoint; `load_data(tier,
    directory)` the data passed to it (None if it builds its own). `module` names the task
    module, whose fingerprint the results carry; `experiment` the training experiment, whose
    fingerprint the checkpoints must carry.
    """

    name: str
    module: str
    experiment: str
    models: tuple[str, ...]
    report_seeds: tuple[int, ...]
    metric: str
    lower_is_better: bool
    results: str
    evaluate: Callable[[Path, torch.device, object, int | None], dict]
    load_data: Callable[[Tier, Path | None], object]


def load_task(name: str) -> Task:
    """The Task of the task module of `name` (imported here, so this module imports no task)."""
    return importlib.import_module(f"ebt.experiments.{TASK_MODULES[name]}").TASK


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Job:
    model: str
    seed: int
    checkpoint: Path
    output: Path

    @property
    def path(self) -> Path:
        return self.output / f"{self.model}-seed{self.seed}.json"


def provenance(job: Job, fingerprint: str, rows: int | None) -> dict:
    """What a result must match to be skipped: the code, the checkpoint and the settings."""
    return {
        "code_fingerprint": fingerprint,
        "checkpoint": {
            "path": recorded_path(job.checkpoint),
            "code_fingerprint": load_checkpoint(job.checkpoint)["code_fingerprint"],
            "sha256": file_sha256(job.checkpoint),
        },
        "test_rows_limit": rows,
    }


def current(stored: dict, wanted: dict) -> bool:
    return all(stored.get(key) == value for key, value in wanted.items())


def execute(
    task: Task,
    models: Sequence[str],
    seeds: Sequence[int],
    checkpoints: Path,
    output: Path,
    device: torch.device,
    data,
    reuse_stale: bool,
    rows: int | None = None,
) -> dict:
    """Evaluate every checkpoint of `models` × `seeds`, then write the summary.

    Before any work: every checkpoint must exist and carry the fingerprint of the current code
    of its experiment, and no existing result may be stale (another fingerprint, checkpoint or
    setting); otherwise runner.StaleResultsError lists them, unless `reuse_stale`. With it, a
    stale result is kept and listed in the summary, and a checkpoint from other code is
    evaluated and marked in its result (checkpoint_from_other_code).
    """
    fingerprint = code_fingerprint(task.module)
    training_fingerprint = code_fingerprint(task.experiment)
    jobs = [
        Job(model, seed, checkpoint_path(checkpoints, model, seed), output)
        for model in models
        for seed in seeds
    ]
    missing = [job.checkpoint for job in jobs if not job.checkpoint.exists()]
    if missing:
        listing = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"{len(missing)} final checkpoints are missing:\n{listing}")
    wanted = {job: provenance(job, fingerprint, rows) for job in jobs}
    old_checkpoints = [
        f"{job.checkpoint} (code {wanted[job]['checkpoint']['code_fingerprint']})"
        for job in jobs
        if wanted[job]["checkpoint"]["code_fingerprint"] != training_fingerprint
    ]
    stale = [
        job for job in jobs if job.path.exists() and not current(read_json(job.path), wanted[job])
    ]
    found = old_checkpoints + [str(job.path) for job in stale]
    if found and not reuse_stale:
        runner.raise_stale(found)
    for job in jobs:
        if job.path.exists() and (job in stale or current(read_json(job.path), wanted[job])):
            continue
        began = time.perf_counter()
        result = task.evaluate(job.checkpoint, device, data, rows)
        payload = {
            "experiment": EXPERIMENT,
            "task": task.name,
            "fingerprint_module": task.module,
            "model": job.model,
            "seed": job.seed,
            "device": device.type,
            "environment": environment(),
            "evaluation_seeds": evaluation_seeds(job.seed),
            "checkpoint_from_other_code": wanted[job]["checkpoint"]["code_fingerprint"]
            != training_fingerprint,
            "seconds": time.perf_counter() - began,
            **wanted[job],
            **result,
        }
        write_json(job.path, payload)
        runner.status(f"{task.name} {job.path.name} ({payload['seconds']:.1f} s)")
    summary = summarize(task, models, seeds, output)
    summary["reused_stale_files"] = sorted(str(job.path) for job in stale)
    summary["code_fingerprint"] = fingerprint
    summary["fingerprint_module"] = task.module
    write_json(output / "summary.json", summary)
    (output / "table.md").write_text(table(task.name, summary))
    return summary


def interval(values: Sequence[float | None]) -> dict:
    """Mean, 95% t half-width and the values over seeds. Undefined values (None or not finite)
    are left out of the mean and counted in `defined`."""
    defined = [v for v in values if v is not None and math.isfinite(v)]
    result = {"values": list(values), "defined": len(defined)}
    if len(defined) >= 2:
        result["mean"], result["half_width"] = mean_interval(defined)
    elif defined:
        result["mean"], result["half_width"] = defined[0], None
    else:
        result["mean"], result["half_width"] = None, None
    if len(defined) >= 2:
        result["std"] = _std(defined)
    return result


G4_SCALARS = ("spearman", "spearman_perturbed_first", "standard_psnr", "perturbed_first_psnr")


def summarize(task: Task, models: Sequence[str], seeds: Sequence[int], output: Path) -> dict:
    """Means over seeds with 95% t-intervals, and the G3 ranges. Every G4 value is averaged
    per seed: a best-of-N fraction is the mean of the seeds' fractions."""
    summary: dict = {"task": task.name, "seeds": list(seeds), "models": {}}
    for model in models:
        results = [read_json(output / f"{model}-seed{seed}.json") for seed in seeds]
        entry: dict = {}
        if "g3" in results[0]:
            per_seed = [r["g3"][task.metric] for r in results]
            entry["g3"] = {
                "metric": task.metric,
                "steps": list(STEP_GRID),
                "per_step": [interval(column) for column in zip(*per_seed, strict=True)],
                "classification": classify_ranges(STEP_GRID, per_seed, task.lower_is_better),
            }
            if "on_branch" in results[0]["g3"]:
                entry["g3"]["on_branch"] = [
                    interval(column)
                    for column in zip(*[r["g3"]["on_branch"] for r in results], strict=True)
                ]
        if "c3" in results[0]:
            settings = []
            for i, setting in enumerate(results[0]["c3"]):
                rows = [r["c3"][i] for r in results]
                settings.append(
                    {
                        "divisor": setting["divisor"],
                        "accept_lower": setting["accept_lower"],
                        "step_size": [row["step_size"] for row in rows],
                        "branch_distance": [
                            interval(c)
                            for c in zip(*[row["branch_distance"] for row in rows], strict=True)
                        ],
                        "on_branch": [
                            interval(c)
                            for c in zip(*[row["on_branch"] for row in rows], strict=True)
                        ],
                    }
                )
            entry["c3"] = settings
        if "g4" in results[0]:
            g4: dict = {"non_finite_values": [r["g4"].get("non_finite_values") for r in results]}
            first = results[0]["g4"]
            for key in G4_SCALARS:
                if key in first:
                    g4[key] = interval([r["g4"][key] for r in results])
            if "best_of_n" in first:
                g4["best_of_n"] = []
                for i, row in enumerate(first["best_of_n"]):
                    rows = [r["g4"]["best_of_n"][i] for r in results]
                    aggregated = {"n": row["n"]}
                    for key in row:
                        if key != "n":
                            aggregated[key] = interval([x[key] for x in rows])
                    g4["best_of_n"].append(aggregated)
            if "coverage" in first:
                g4["coverage"] = [
                    {
                        "n": row["n"],
                        "coverage": interval([r["g4"]["coverage"][i]["coverage"] for r in results]),
                    }
                    for i, row in enumerate(first["coverage"])
                ]
            entry["g4"] = g4
        summary["models"][model] = entry
    return summary


def shown(value: dict, digits: int = 4) -> str:
    mean, half = value["mean"], value["half_width"]
    if mean is None:
        return "undefined"
    text = f"{mean:.{digits}f}" + ("" if half is None else f" ± {half:.{digits}f}")
    total = len(value["values"])
    if value["defined"] < total:
        text += f" ({value['defined']} of {total} defined)"
    return text


G4_LABELS = {
    "spearman": "Spearman ρ(final energy, error)",
    "spearman_perturbed_first": "Spearman ρ, first perturbed candidate",
    "standard_psnr": "PSNR, standard output",
    "perturbed_first_psnr": "PSNR, first perturbed candidate (N = 1)",
}


def ranges_text(classification: dict) -> str:
    parts = []
    for kind in ("falling", "flat", "rising", "undefined"):
        found = classification[kind]
        where = found if found == "absent" else ", ".join(f"{a}–{b}" for a, b in found)
        parts.append(f"{kind} {where}")
    return "; ".join(parts)


def table(task: str, summary: dict) -> str:
    """The summary as Markdown tables: means over seeds ± half-width of the 95% t-interval."""
    lines = [
        f"# G3 and G4, {task}",
        "",
        f"Mean over seeds {', '.join(map(str, summary['seeds']))} ± half-width of the 95% "
        "t-interval (method.md section 19). Undefined seeds are left out of a mean and counted "
        "as (k of n defined).",
    ]
    for model, entry in summary["models"].items():
        lines += ["", f"## {model}"]
        if "g3" in entry:
            g3 = entry["g3"]
            lines += ["", f"G3, test {g3['metric']} against inference steps:", ""]
            lines += ["| steps | " + g3["metric"] + " | std over seeds |", "|---|---|---|"]
            for k, value in zip(g3["steps"], g3["per_step"], strict=True):
                std = value.get("std")
                lines.append(f"| {k} | {shown(value)} | {'' if std is None else f'{std:.4f}'} |")
            ranges = g3["classification"]["ranges"]
            order = " → ".join(f"{r['kind']} {r['from']}–{r['to']}" for r in ranges)
            lines += ["", f"Ranges: {ranges_text(g3['classification'])}.", f"In order: {order}."]
        if "c3" in entry:
            lines += ["", "G3, (c3) inference experiment, test branch distance r and fraction "
                      "with r ≤ 0.1:", ""]  # fmt: skip
            lines += ["| α divisor | accept-only-lower | " + " | ".join(
                f"r, {k} steps" for k in STEP_GRID) + " | on branch, 64 steps |"]  # fmt: skip
            lines.append("|---|---|" + "---|" * (len(STEP_GRID) + 1))
            for setting in entry["c3"]:
                cells = [shown(v, 3) for v in setting["branch_distance"]]
                lines.append(
                    f"| {setting['divisor']} | {'yes' if setting['accept_lower'] else 'no'} | "
                    + " | ".join(cells)
                    + f" | {shown(setting['on_branch'][-1], 3)} |"
                )
        if "g4" in entry:
            g4 = entry["g4"]
            lines.append("")
            bad = [n for n in g4.get("non_finite_values", []) if n]
            if bad:
                lines.append(
                    f"- G4 undefined for {len(bad)} seeds (non-finite values: "
                    f"{', '.join(str(n) for n in g4['non_finite_values'])})"
                )
            for key, label in G4_LABELS.items():
                if key in g4:
                    lines.append(f"- {label}: {shown(g4[key])}")
            if "best_of_n" in g4:
                lines += ["", "| N | random | lowest energy | oracle | fraction |",
                          "|---|---|---|---|---|"]  # fmt: skip
                for row in g4["best_of_n"]:
                    lines.append(
                        f"| {row['n']} | {shown(row['random'])} | {shown(row['lowest_energy'])} | "
                        f"{shown(row['oracle'])} | {shown(row['fraction'], 3)} |"
                    )
            if "coverage" in g4:
                lines += ["", "Both-branch coverage: " + "; ".join(
                    f"N = {row['n']}: {shown(row['coverage'], 3)}" for row in g4["coverage"]
                )]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=list(TASK_MODULES), required=True)
    parser.add_argument("--tier", choices=[tier.value for tier in Tier], required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None, help="evaluate the first test rows only")
    parser.add_argument("--reuse-stale", action="store_true", help=runner.REUSE_HELP)
    args = parser.parse_args()
    task, tier = load_task(args.task), Tier(args.tier)
    models = args.models or list(task.models)
    seeds = args.seeds or list(task.report_seeds)
    data = task.load_data(tier, args.data)
    checkpoints = args.checkpoints or checkpoint_directory(task.experiment, tier.value)
    output = args.output or Path(task.results) / "g3_g4" / tier.value
    execute(task, models, seeds, checkpoints, output, torch.device(args.device), data,
            args.reuse_stale, args.rows)  # fmt: skip


if __name__ == "__main__":
    main()
