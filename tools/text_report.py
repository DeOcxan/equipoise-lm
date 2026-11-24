"""Table, samples and validation plot of the text task, generated from saved result files only.

usage: python -m tools.text_report RESULTS_DIR

Reads the per-seed result files of RESULTS_DIR (results/text/lm/full for v1.0) and its
search.json, if present.

Outputs in RESULTS_DIR: table.md, samples.md and validation.png.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebt.experiments.text_lm import MODELS
from ebt.results import RunResult, read_json, read_result
from ebt.statistics import mean_interval
from tools.palette import COLORS as TASK_COLORS
from tools.palette import GRID, SURFACE, TEXT

LABELS = {"ebt": "EBT, S2, 4 inference steps", "transformer": "Causal transformer"}
COLORS = TASK_COLORS["text"]


def load(directory: Path) -> dict[str, list[RunResult]]:
    runs = {model: [] for model in MODELS}
    for path in sorted(directory.glob("*-seed*.json")):
        result = read_result(path)
        runs[result.model].append(result)
    return {model: sorted(r, key=lambda r: r.seed) for model, r in runs.items() if r}


def interval(values: list[float], digits: int = 3) -> str:
    if len(values) < 2:
        return f"{values[0]:.{digits}f} (one seed)"
    mean, half = mean_interval(values)
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


PASS_KINDS = ("forward", "backward", "second_order")


def passes(counts: dict, examples: int = 1) -> str:
    return " / ".join(f"{counts[key] / examples:g}" for key in PASS_KINDS)


def training_passes(results: list[RunResult]) -> str:
    """Training passes per fresh window: for each kind the mean over the seeds and, where the
    seeds differ, the per-seed range (the EBT's step count varies with its seed)."""
    cells = []
    for key in PASS_KINDS:
        values = [
            r.cost["train_total_example_passes"][key] / r.cost["train_examples"] for r in results
        ]
        mean = sum(values) / len(values)
        if min(values) == max(values):
            cells.append(f"{values[0]:g}")
        else:
            cells.append(f"{mean:.5f} ({min(values):.5f} to {max(values):.5f})")
    return " / ".join(cells)


def metric(results: list[RunResult], split: str) -> str:
    values = [r.metrics[split]["bpc"] for r in results if split in r.metrics]
    return interval(values) if values else "n/a"


def table(runs: dict[str, list[RunResult]], search: dict | None) -> str:
    seeds = sorted({len(results) for results in runs.values()})
    first = next(iter(runs.values()))[0]
    vocabulary = first.config["model"]["vocabulary"]
    lines = [
        f"Bits per character at each run's best checkpoint (lowest validation BPC); mean over "
        f"{', '.join(map(str, seeds))} seeds ± half-width of the 95% t-interval. Uniform "
        f"prediction: log2 {vocabulary} = {math.log2(vocabulary):.3f}. Passes per window: "
        "forward / first-order backward / second-order (method.md section 18.6); training "
        "passes per fresh window, the EBT's replayed rows included, as the mean over seeds with "
        "the per-seed range where the seeds differ.",
        "",
        "| Model | test BPC | out-of-distribution BPC | validation BPC | best step | "
        "parameters | training passes | inference passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, results in runs.items():
        head = results[0]
        best_steps = ", ".join(str(r.history["best_step"]) for r in results)
        lines.append(
            f"| {LABELS[model]} | {metric(results, 'test')} | {metric(results, 'ood')} | "
            f"{metric(results, 'validation')} | {best_steps} | "
            f"{head.parameters['trainable']:,} | "
            f"{training_passes(results)} | "
            f"{passes(head.cost['inference_example_passes_per_example'])} |"
        )
    ood = first.config.get("ood_text")
    if ood is not None:
        lines += [
            "",
            f"Out-of-distribution text: {ood['block_characters']:,} characters of Pride and "
            f"Prejudice after the rule of method.md section 18.1; "
            f"{ood['dropped']:,} of {ood['novel_characters']:,} characters of the novel "
            f"({100 * ood['dropped_fraction']:.2f}%) lay outside the vocabulary and were "
            "dropped.",
        ]
    if search is not None:
        lines += ["", *chosen(search)]
    return "\n".join(lines) + "\n"


def chosen(search: dict) -> list[str]:
    lines = ["Search (seed 100, validation BPC at the best checkpoint), in grid order:", ""]
    for model, record in search["models"].items():
        for entry in record["tried"]:
            settings = ", ".join(
                f"{key} {value}"
                for key, value in sorted(entry["candidate"].items())
                if key != "model" and value is not None
            )
            value = entry["selection_value"]
            shown = "undefined" if value is None else f"{value:.4f}"
            mark = " (chosen)" if entry["candidate"] == record["chosen"] else ""
            lines.append(f"- {LABELS[model]}: {settings}; {shown}{mark}")
    return lines


def samples(runs: dict[str, list[RunResult]]) -> str:
    lines = ["Generated text, seed 0, temperature 1 (illustration, not a metric).", ""]
    for model, results in runs.items():
        artifacts = results[0].artifacts.get("samples")
        if artifacts is None:
            continue
        lines += [f"## {LABELS[model]}", "", "Prompt:", "", "```", artifacts["prompt"], "```"]
        for i, text in enumerate(artifacts["texts"]):
            lines += ["", f"Sample {i + 1}:", "", "```", text, "```"]
        lines.append("")
    return "\n".join(lines)


def plot_validation(runs: dict[str, list[RunResult]], path: Path) -> None:
    """Validation BPC against the training step, one thin line per seed."""
    figure, axis = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for model, results in runs.items():
        for i, r in enumerate(results):
            axis.plot(
                r.history["validation_steps"],
                r.history["validation_bpc"],
                color=COLORS[model],
                linewidth=1.5,
                label=LABELS[model] if i == 0 else None,
            )
    axis.set_xlabel("training step")
    axis.set_ylabel("validation bits per character")
    axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Validation curves, one line per seed", color=TEXT)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    runs = load(directory)
    summary = directory / "search.json"
    text = table(runs, read_json(summary) if summary.exists() else None)
    (directory / "table.md").write_text(text)
    (directory / "samples.md").write_text(samples(runs))
    print(text)
    plot_validation(runs, directory / "validation.png")


if __name__ == "__main__":
    main()
