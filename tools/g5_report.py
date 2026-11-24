"""Question G5: relative improvement from thinking in and out of distribution (method.md 20).

usage: python -m tools.g5_report [--tier full] [--results results] [--tasks regression mnist text]

For every EBT checkpoint of a task, pairs the in-distribution G3 result
(<results>/<task>/g3_g4/<tier>/) with the out-of-distribution one (<results>/<task>/g5/<tier>/,
for MNIST one subdirectory noise-<level> per level), after checking the pair (`check_pair`). Then
computes I = 1 - e_64 / e_1 per checkpoint (`improvement`), Δ = I_OOD - I_ID per seed, their means
and 95% t-intervals, and the outcome of method.md 20.2 (`outcome`).

Outputs: <results>/<task>/g5/<tier>/g5.json and table.md.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ebt.experiments.inference_g3_g4 import STEP_GRID, interval
from ebt.experiments.inference_g5 import (
    EXPERIMENT,
    MNIST_LEVELS,
    PRIMARY_LEVEL,
    TASKS,
    level_directory,
)
from ebt.fingerprint import code_fingerprint
from ebt.results import read_json, write_json

FIRST, LAST = STEP_GRID[0], STEP_GRID[-1]
"""F_0 = 1 and F = 64 inference steps (method.md 20.2)."""

NOT_THE_PAPER = " This is not the paper's effect."
"""Added to outcomes 3, 4 and 5 for the text task only, for the reason given at PAPER_EFFECT."""
PAPER_EFFECT = " This is the effect the paper reports for language (§4.1.3)."
"""Added to outcome 2 for the text task only: the paper's claim is for language; §4.3 makes no
such claim for images, and the regression task has no counterpart in the paper."""
PAPER_EFFECT_OOD = (
    " Out of distribution, this matches the effect the paper reports for language (§4.1.3)."
)
"""Added instead of PAPER_EFFECT to outcome 2 for the text task when the mean of I_ID is below 0:
the paper's effect then holds out of distribution only."""
OUTCOMES = {
    "undefined": "Undefined: I_ID, I_OOD or Δ has fewer than 2 defined seeds.",
    "helps_more": "Thinking helps more out of distribution.",
    "degrades_ood_only": "Thinking degrades the result out of distribution only.",
    "difference_only": "The difference is above 0 but the out-of-distribution improvement is "
    "not measurably above 0.",
    "helps_less": "Thinking helps less out of distribution.",
    "no_difference": "No measurable difference.",
}
"""The basic texts of the outcomes of method.md 20.2 other than outcome 3, in the order the rule
tests them; `outcome_text` gives the text a report writes."""

HELPS_OOD_DEGRADES_ID = (
    "Thinking helps out of distribution and degrades the result in distribution; Δ therefore "
    "includes the in-distribution degradation."
)
"""Outcome 2 when the mean of I_ID is below 0."""
NO_HELP_OOD = (
    "Thinking does not help out of distribution on average; its effect there is smaller than in "
    "distribution."
)
"""Outcome 6 when the mean of I_OOD is below 0."""
DEGRADATION = {
    "degrades_both_smaller": "smaller",
    "degrades_both_larger": "larger",
    "degrades_both_same": "not measurably different",
}
"""The three keys of outcome 3 and the relative degradation each states; `outcome_text` composes
their texts."""


class G5ReportError(Exception):
    """A required G5 file is missing, stale, limited, of another split or level, or from
    another checkpoint than its in-distribution counterpart."""


def error_of(metric: str, value: float) -> float:
    """The error whose ratio defines I, from the value a G3 curve records: r itself, the
    perplexity 2^BPC, or the MSE 255**2 · 10^(-PSNR / 10) behind the PSNR."""
    if metric == "branch_distance":
        return value
    if metric == "bpc":
        return 2.0**value
    if metric == "psnr":
        return 255.0**2 * 10.0 ** (-value / 10)
    raise ValueError(f"unknown metric {metric!r}")


def improvement(metric: str, first: float | None, last: float | None) -> float | None:
    """I = 1 - e(last) / e(first); None when a value is undefined or e(first) is 0."""
    if first is None or last is None or not math.isfinite(first) or not math.isfinite(last):
        return None
    before, after = error_of(metric, first), error_of(metric, last)
    if before == 0:
        return None
    return 1 - after / before


def at_step(g3: dict, metric: str, step: int) -> float | None:
    return dict(zip(g3["steps"], g3[metric], strict=True))[step]


def curve_improvement(metric: str, g3: dict) -> float | None:
    """I of one G3 curve, from its values at FIRST and LAST steps only."""
    return improvement(metric, at_step(g3, metric, FIRST), at_step(g3, metric, LAST))


def differences(
    inside: Sequence[float | None], outside: Sequence[float | None]
) -> list[float | None]:
    """Δ(s) = I_OOD(s) - I_ID(s) per seed; None where either is undefined."""
    return [None if a is None or b is None else b - a for a, b in zip(inside, outside, strict=True)]


def above(summary: dict) -> bool:
    """The interval lies above 0: its lower end is above 0 (an end at 0 is not)."""
    mean, half = summary["mean"], summary["half_width"]
    return mean is not None and half is not None and mean - half > 0


def below(summary: dict) -> bool:
    """The interval lies below 0: its upper end is below 0 (an end at 0 is not)."""
    mean, half = summary["mean"], summary["half_width"]
    return mean is not None and half is not None and mean + half < 0


def outcome(i_id: dict, i_ood: dict, delta: dict) -> str:
    """The outcome of method.md 20.2 (a key of OUTCOMES or DEGRADATION), the first rule that
    applies."""
    if any(s["mean"] is None or s["half_width"] is None for s in (i_id, i_ood, delta)):
        return "undefined"
    if above(delta) and above(i_ood):
        return "helps_more"
    if i_id["mean"] < 0 and (below(i_ood) or i_ood["mean"] < 0):
        if above(delta):
            return "degrades_both_smaller"
        if below(delta):
            return "degrades_both_larger"
        return "degrades_both_same"
    if below(i_ood):
        return "degrades_ood_only"
    if above(delta):
        return "difference_only"
    if below(delta):
        return "helps_less"
    return "no_difference"


def outcome_text(key: str, task: str, i_id: dict, i_ood: dict) -> str:
    """The text of outcome `key` for `task` (method.md 20.2). Outcomes 2 to 5 refer to the paper
    for text only; outcome 2 states an in-distribution degradation; outcome 6 states when
    thinking does not help out of distribution on average; outcome 3 says "on average" for a side
    whose interval is not below 0."""
    if key == "helps_more":
        if i_id["mean"] < 0:
            return HELPS_OOD_DEGRADES_ID + (PAPER_EFFECT_OOD if task == "text" else "")
        return OUTCOMES[key] + (PAPER_EFFECT if task == "text" else "")
    if key == "helps_less" and i_ood["mean"] < 0:
        return NO_HELP_OOD
    not_the_paper = NOT_THE_PAPER if task == "text" else ""
    if key in DEGRADATION:
        if below(i_id) and below(i_ood):
            sides = "in both"
        else:
            inside = "" if below(i_id) else " on average"
            outside = "" if below(i_ood) else " on average"
            sides = f"in distribution{inside} and out of distribution{outside}"
        return (
            f"Thinking degrades the result {sides}; the relative degradation out of distribution "
            f"is {DEGRADATION[key]}." + not_the_paper
        )
    if key in ("degrades_ood_only", "difference_only"):
        return OUTCOMES[key] + not_the_paper
    return OUTCOMES[key]


def expected_settings(task: str, level: float | None) -> dict:
    """The settings a G5 result of `task` (and MNIST `level`) must record (inference_g5)."""
    if task == "mnist":
        return {"name": "test", "noise_level": level, "noise_seed": MNIST_LEVELS[level]}
    return {"name": "ood"}


def field(data: dict, path: Path, keys: tuple[str, ...], label: str) -> Any:
    """data[keys[0]][keys[1]]...; G5ReportError naming the file and the field if absent."""
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise G5ReportError(f"{label}: {path} has no field {'.'.join(keys)}")
        value = value[key]
    return value


def require_result(data: dict, path: Path, label: str) -> None:
    """The fields the report reads from a G3 result besides its curve: seed, checkpoint.sha256
    and the trained step count."""
    field(data, path, ("seed",), label)
    field(data, path, ("checkpoint", "sha256"), label)
    field(data, path, ("trained_inference", "steps"), label)


def require_curve(data: dict, path: Path, metric: str, label: str) -> None:
    """A G3 curve of `metric` with a value for every step of the grid."""
    steps = field(data, path, ("g3", "steps"), label)
    values = field(data, path, ("g3", metric), label)
    if len(values) != len(steps):
        raise G5ReportError(f"{label}: {path} field g3.{metric} does not match g3.steps")
    for step in STEP_GRID:
        if step not in steps:
            raise G5ReportError(f"{label}: {path} has no field g3 step {step}")


def check_pair(
    inside_path: Path, outside_path: Path, settings: dict, label: str, metric: str
) -> tuple:
    """Read and check one in- and out-of-distribution pair; G5ReportError, naming the file and
    any missing field, on the first problem.

    Both files must exist. The out-of-distribution result must carry the current inference_g5
    fingerprint, have no row limit and the settings of its split and level. Both must have every
    field the report reads: the seed, the checkpoint, the trained step count and a G3 curve with
    every step of the grid. Both must come from the same checkpoint (equal SHA-256)."""
    for path in (inside_path, outside_path):
        if not path.exists():
            raise G5ReportError(f"{label}: missing {path}")
    inside, outside = read_json(inside_path), read_json(outside_path)
    if outside.get("code_fingerprint") != code_fingerprint(EXPERIMENT):
        raise G5ReportError(f"{label}: {outside_path} is not from the current {EXPERIMENT} code")
    if outside.get("test_rows_limit") is not None:
        raise G5ReportError(
            f"{label}: {outside_path} is limited to {outside['test_rows_limit']} rows"
        )
    if outside.get("settings") != settings:
        raise G5ReportError(
            f"{label}: {outside_path} has settings {outside.get('settings')}, expected {settings}"
        )
    require_result(inside, inside_path, label)
    require_curve(inside, inside_path, metric, label)
    require_result(outside, outside_path, label)
    require_curve(outside, outside_path, metric, label)
    if inside["checkpoint"]["sha256"] != outside["checkpoint"]["sha256"]:
        raise G5ReportError(
            f"{label}: {inside_path} and {outside_path} come from different checkpoints"
        )
    return inside, outside


def compare(metric: str, inside: list[dict], outside: list[dict], task: str = "") -> dict:
    """I_ID, I_OOD, Δ and the outcome of one model from its per-seed results (same seeds)."""
    i_id = [curve_improvement(metric, r["g3"]) for r in inside]
    i_ood = [curve_improvement(metric, r["g3"]) for r in outside]
    trained = inside[0]["trained_inference"]["steps"]
    result = {
        "seeds": [r["seed"] for r in inside],
        "trained_steps": trained,
        "i_id": interval(i_id),
        "i_ood": interval(i_ood),
        "delta": interval(differences(i_id, i_ood)),
    }
    for name, results in (("id", inside), ("ood", outside)):
        for step in (FIRST, trained, LAST):
            values = [at_step(r["g3"], metric, step) for r in results]
            result[f"{metric}_{name}_{step}"] = interval(values)
    result["outcome"] = outcome(result["i_id"], result["i_ood"], result["delta"])
    result["outcome_text"] = outcome_text(result["outcome"], task, result["i_id"], result["i_ood"])
    return result


def task_report(results: Path, task: str, tier: str) -> dict:
    """The comparison of every model (and MNIST level) of `task`; G5ReportError if any
    required file is missing or does not match."""
    settings = TASKS[task]
    base = results / task / "g5" / tier
    inside_dir = results / task / "g3_g4" / tier
    levels = sorted(MNIST_LEVELS) if task == "mnist" else [None]
    report: dict = {"task": task, "tier": tier, "metric": settings.metric, "comparisons": []}
    for level in levels:
        directory = base if level is None else level_directory(base, level)
        expected = expected_settings(task, level)
        for model in settings.models:
            pairs = []
            for seed in settings.report_seeds:
                where = "" if level is None else f" level {level:g}"
                label = f"{task} {model}{where} seed {seed}"
                pairs.append(
                    check_pair(
                        inside_dir / f"{model}-seed{seed}.json",
                        directory / f"{model}-seed{seed}.json",
                        expected,
                        label,
                        settings.metric,
                    )
                )
            primary = level in (None, PRIMARY_LEVEL)
            entry = {"model": model, "noise_level": level, "primary": primary}
            inside, outside = [p[0] for p in pairs], [p[1] for p in pairs]
            entry.update(compare(settings.metric, inside, outside, task))
            if not primary:
                entry["outcome"] = None  # reported without an outcome (method.md 20.2)
                entry["outcome_text"] = None
            report["comparisons"].append(entry)
    return report


def shown(value: dict, digits: int = 4) -> str:
    mean, half = value["mean"], value["half_width"]
    if mean is None:
        return "undefined"
    text = f"{mean:.{digits}f}" + ("" if half is None else f" ± {half:.{digits}f}")
    if value["defined"] < len(value["values"]):
        text += f" ({value['defined']} of {len(value['values'])} defined)"
    return text


def where_of(entry: dict) -> str:
    if entry["noise_level"] is None:
        return "split"
    where = f"noise {entry['noise_level']:g}"
    return where if entry["primary"] else where + " (no decision)"


ERRORS = {
    "branch_distance": "branch distance r",
    "psnr": "MSE behind the PSNR",
    "bpc": "perplexity 2^BPC",
}


def table(report: dict) -> str:
    metric = report["metric"]
    lines = [
        f"# G5, {report['task']}",
        "",
        f"I = 1 − e₆₄ / e₁ per checkpoint (method.md section 20.2; e: {ERRORS[metric]}); "
        "Δ = I_OOD − I_ID per seed. Mean over seeds ± half-width of the 95% t-interval. "
        "I lies in (−∞, 1]. The outcome follows the rule of method.md section 20.2.",
        "",
        "| model | out of distribution | I_ID | I_OOD | Δ | outcome |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report["comparisons"]:
        lines.append(
            f"| {entry['model']} | {where_of(entry)} | {shown(entry['i_id'])} | "
            f"{shown(entry['i_ood'])} | {shown(entry['delta'])} | "
            f"{entry['outcome_text'] or '-'} |"
        )
    lines += [
        "",
        f"{metric} at 1 step, at the trained step count and at 64 steps:",
        "",
        "| model | out of distribution | data | 1 | trained | 64 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in report["comparisons"]:
        trained = entry["trained_steps"]
        for name, label in (("id", "in distribution"), ("ood", "out of distribution")):
            cells = [shown(entry[f"{metric}_{name}_{step}"]) for step in (FIRST, trained, LAST)]
            lines.append(
                f"| {entry['model']} | {where_of(entry)} | {label} | {cells[0]} | {cells[1]} "
                f"({trained} steps) | {cells[2]} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", default="full")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    args = parser.parse_args()
    for task in args.tasks:
        try:
            report = task_report(args.results, task, args.tier)
        except G5ReportError as error:
            print(f"G5 report error: {error}", file=sys.stderr)
            return 1
        directory = args.results / task / "g5" / args.tier
        write_json(directory / "g5.json", report)
        (directory / "table.md").write_text(table(report))
        print(table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
