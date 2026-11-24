"""Table and plots of the MNIST denoising task, generated from saved result files only.

usage: python -m tools.mnist_report RESULTS_DIR

Reads the per-seed result files of RESULTS_DIR (results/mnist/denoising/full for v1.0).

Outputs in RESULTS_DIR: table.md, examples.png and validation.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ebt.experiments.mnist_denoising import MODELS
from ebt.results import RunResult, read_json, read_result
from ebt.statistics import mean_interval
from tools.palette import COLORS as TASK_COLORS
from tools.palette import GRID, SURFACE, TEXT

LABELS = {"ebt": "EBT, S1, 2 inference steps", "denoiser": "Feed-forward denoiser"}
COLORS = TASK_COLORS["mnist"]
TAIL_STEPS = 2_000
"""Length of the end of training over which the validation curve's change is reported."""


def load(directory: Path) -> dict[str, list[RunResult]]:
    runs = {model: [] for model in MODELS}
    for path in sorted(directory.glob("*-seed*.json")):
        result = read_result(path)
        runs[result.model].append(result)
    return {model: sorted(r, key=lambda r: r.seed) for model, r in runs.items() if r}


def interval(values: list[float], digits: int = 3) -> str:
    mean, half = mean_interval(values)
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def passes(counts: dict, examples: int = 1) -> str:
    return " / ".join(
        f"{counts[key] / examples:g}" for key in ("forward", "backward", "second_order")
    )


def table(runs: dict[str, list[RunResult]], search: dict | None) -> str:
    seeds = sorted({len(results) for results in runs.values()})
    lines = [
        f"Test split, 10 000 images, noise level 0.1; mean over {', '.join(map(str, seeds))} "
        "seeds ± half-width of the 95% t-interval. PSNR from one MSE over all test pixels "
        "(method.md section 17.6). Passes per example: forward / first-order backward / "
        "second-order.",
        "",
        "| Model | test PSNR (dB) | test Smooth L1 | test MSE | parameters | training passes | "
        "inference passes |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, results in runs.items():
        first = results[0]
        lines.append(
            f"| {LABELS[model]} | {interval([r.metrics['test']['psnr'] for r in results], 2)} | "
            f"{interval([r.metrics['test']['smooth_l1'] for r in results], 5)} | "
            f"{interval([r.metrics['test']['mse'] for r in results], 5)} | "
            f"{first.parameters['trainable']:,} | "
            f"{passes(first.cost['train_total_example_passes'], first.cost['train_examples'])} | "
            f"{passes(first.cost['inference_example_passes_per_example'])} |"
        )
    noisy = next(iter(runs.values()))[0].metrics["test"]["noisy_input_psnr"]
    lines.append(f"| Noisy input | {noisy:.2f} (fixed noise, no spread) | | | | | |")

    if "ebt" in runs:
        alphas = [r.history["final_step_size"] for r in runs["ebt"]]
        lines += ["", f"EBT step size α at the end of training: {interval(alphas, 1)}."]
    lines += ["", *curve_summary(runs)]
    if search is not None:
        lines += ["", *chosen(search)]
    return "\n".join(lines) + "\n"


def curve_summary(runs: dict[str, list[RunResult]]) -> list[str]:
    """Whether the validation loss was still falling at the end of training, per model."""
    lines = [
        f"Change of the validation Smooth L1 over the last {TAIL_STEPS} training steps and",
        "between the last two evaluations (negative: still falling):",
        "",
        "| Model | last steps | last two evaluations | seeds still falling |",
        "|---|---|---|---|",
    ]
    for model, results in runs.items():
        tail, last = [], []
        for r in results:
            steps, values = r.history["validation_steps"], r.history["validation_smooth_l1"]
            start = next(i for i, step in enumerate(steps) if step >= steps[-1] - TAIL_STEPS)
            tail.append(values[-1] - values[start])
            last.append(values[-1] - values[-2])
        falling = sum(change < 0 for change in last)
        lines.append(
            f"| {LABELS[model]} | {interval(tail, 6)} | {interval(last, 6)} | "
            f"{falling} of {len(results)} |"
        )
    return lines


def chosen(search: dict) -> list[str]:
    lines = ["Chosen configurations (search seed 100, validation Smooth L1 after inference):", ""]
    for model, record in search["models"].items():
        best = record["chosen"]
        value = next(e for e in record["tried"] if e["candidate"] == best)["selection_value"]
        settings = ", ".join(
            f"{key} {value}"
            for key, value in sorted(best.items())
            if key != "model" and value is not None
        )
        lines.append(f"- {LABELS[model]}: {settings}; validation Smooth L1 {value:.6f}.")
    return lines


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot_validation(runs: dict[str, list[RunResult]], path: Path) -> None:
    """Validation PSNR and Smooth L1 against the training step, one thin line per seed."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE)
    for axis, key, label in zip(
        axes, ("validation_psnr", "validation_smooth_l1"), ("PSNR (dB)", "Smooth L1"), strict=True
    ):
        style(axis)
        for model, results in runs.items():
            for i, r in enumerate(results):
                axis.plot(
                    r.history["validation_steps"],
                    r.history[key],
                    color=COLORS[model],
                    linewidth=1.5,
                    label=LABELS[model] if i == 0 else None,
                )
        axis.set_xlabel("training step")
        axis.set_ylabel(f"validation {label}")
        axis.legend(frameon=False, fontsize=9)
    axes[1].set_yscale("log")
    figure.suptitle("Validation curves, every 500 steps, one line per seed", color=TEXT)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_examples(runs: dict[str, list[RunResult]], path: Path) -> None:
    """Clean, noisy and denoised test images of seed 0: one row each, one column per image."""
    first = next(iter(runs.values()))[0].artifacts["examples"]
    rows = [("clean", first["clean"]), ("noisy input", first["noisy"])]
    for model, results in runs.items():
        outputs = results[0].artifacts["examples"]["outputs"]
        if len(outputs) == 1:
            rows.append((LABELS[model], outputs[0]))
        else:
            rows += [(f"EBT after step {i + 1}", images) for i, images in enumerate(outputs)]
    columns = len(first["clean"])
    figure, axes = plt.subplots(
        len(rows), columns, figsize=(1.4 * columns + 1.6, 1.45 * len(rows)), facecolor=SURFACE
    )
    for row, (label, images) in zip(axes, rows, strict=True):
        for axis, image in zip(row, images, strict=True):
            pixels = np.clip(np.asarray(image) * 0.5 + 0.5, 0, 1)
            axis.imshow(pixels, cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
        row[0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=TEXT)
    figure.suptitle("Test images, seed 0", color=TEXT)
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
    print(text)
    plot_validation(runs, directory / "validation.png")
    plot_examples(runs, directory / "examples.png")


if __name__ == "__main__":
    main()
