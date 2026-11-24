"""Question G3: error against inference steps on the three tasks, one figure from saved results.

usage: python -m tools.g3_figure [--results results] [--output results/g3_steps.png]

Reads results/<task>/g3_g4/full/summary.json and the trained step counts of the per-seed files
next to it (`curves`), and draws one panel per task (`plot`).

Output: the PNG file --output, by default <results>/g3_steps.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebt.results import read_json
from tools.palette import COLORS, GRID, MUTED, SURFACE, TEXT

TASKS = ("text", "mnist", "regression")
MODELS = {
    "text": ("ebt",),
    "mnist": ("ebt",),
    "regression": ("ebt_mse", "ebt_contrastive", "ebt_combined", "ebt_regularized"),
}
LABELS = {
    ("text", "ebt"): "text EBT, S2",
    ("mnist", "ebt"): "MNIST EBT, S1",
    ("regression", "ebt_mse"): "(b) EBT, MSE",
    ("regression", "ebt_contrastive"): "(c1) EBT, contrastive",
    ("regression", "ebt_combined"): "(c2) EBT, MSE + contrastive",
    ("regression", "ebt_regularized"): "(c3) EBT, contrastive + energy penalty",
}
TITLES = {
    "text": "Text EBT (S2), Tiny Shakespeare test split",
    "mnist": "MNIST EBT (S1), noise 0.1, test split",
    "regression": "Regression y = ±√x, test split",
}
Y_LABELS = {
    "text": "bits per character (lower is better)",
    "mnist": "PSNR in dB (higher is better)",
    "regression": "branch distance r, log scale (lower is better)",
}
DASHED = {("regression", "ebt_combined")}
"""Drawn dashed: (c2) lies on (b) at every step count and would otherwise hide it."""
REFERENCES = ((1.0, "r = 1: between the branches"), (0.1, "r = 0.1: on a branch"))
"""Regression reference lines: r = 1 at the conditional mean 0 and the on-branch threshold
r <= 0.1 (method.md 16.7)."""
METRICS = {"text": "bpc", "mnist": "psnr", "regression": "branch_distance"}
FONTS = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}
"""Font sizes in points, chosen so that the figure stays readable at about 900 px wide."""
ANNOTATION_SIZE = 11


class G3FigureError(Exception):
    """A summary or per-seed file is missing a field or disagrees with the others."""


def trained_steps(directory: Path, model: str) -> int:
    """The trained step count shared by every seed of `model` in `directory`."""
    steps = {
        read_json(path)["trained_inference"]["steps"]
        for path in sorted(directory.glob(f"{model}-seed*.json"))
    }
    if len(steps) != 1:
        raise G3FigureError(f"{directory}: trained step counts of {model}: {sorted(steps)}")
    return steps.pop()


def curves(results: Path) -> dict[str, list[dict]]:
    """Per task, one entry per model: steps, mean, lower and upper band, trained step count."""
    out = {}
    for task in TASKS:
        directory = results / task / "g3_g4" / "full"
        summary = read_json(directory / "summary.json")
        entries = []
        for model in MODELS[task]:
            g3 = summary["models"][model]["g3"]
            if g3["metric"] != METRICS[task]:
                raise G3FigureError(f"{directory}: {model} G3 metric is {g3['metric']}")
            rows = g3["per_step"]
            entries.append(
                {
                    "model": model,
                    "steps": list(g3["steps"]),
                    "mean": [row["mean"] for row in rows],
                    "lower": [row["mean"] - row["half_width"] for row in rows],
                    "upper": [row["mean"] + row["half_width"] for row in rows],
                    "trained": trained_steps(directory, model),
                }
            )
        out[task] = entries
    return out


def plot(data: dict[str, list[dict]], path: Path) -> None:
    """One panel per task: the mean over seeds at 1 to 64 inference steps on a log2 axis, with the
    95% t-interval over seeds as a band and the trained step count as a dotted vertical line.

    Text: bits per character (lower is better); MNIST: PSNR (higher is better). Regression: branch
    distance r of (b), (c1), (c2) and (c3) on a log scale, since (c1) lies more than 50 times above
    (b) and (c2), with dashed reference lines at r = 1 and r = 0.1. (c2) is dashed because it lies
    on (b). The output is byte-identical for the same input.
    """
    with plt.rc_context(FONTS):
        draw(data, path)


def draw(data: dict[str, list[dict]], path: Path) -> None:
    figure, axes = plt.subplots(1, len(TASKS), figsize=(16, 5.4), facecolor=SURFACE)
    for axis, task in zip(axes, TASKS, strict=True):
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        trained = set()
        for entry in data[task]:
            key = (task, entry["model"])
            axis.fill_between(
                entry["steps"],
                entry["lower"],
                entry["upper"],
                color=COLORS[task][entry["model"]],
                alpha=0.18,
                linewidth=0,
            )
            axis.plot(
                entry["steps"],
                entry["mean"],
                color=COLORS[task][entry["model"]],
                linewidth=2,
                marker="o",
                markersize=4,
                label=LABELS[key],
                linestyle="--" if key in DASHED else "-",
            )
            trained.add(entry["trained"])
        for steps in sorted(trained):
            axis.axvline(steps, color=MUTED, linewidth=1, linestyle=":")
            axis.annotate(
                f"trained: {steps} steps",
                xy=(steps, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -12),
                textcoords="offset points",
                color=MUTED,
                fontsize=ANNOTATION_SIZE,
            )
        axis.set_xscale("log", base=2)
        steps = data[task][0]["steps"]
        axis.set_xticks(steps, [str(s) for s in steps])
        axis.minorticks_off()
        if task == "regression":
            axis.set_yscale("log")
            for value, label in REFERENCES:
                axis.axhline(value, color=MUTED, linewidth=0.8, linestyle="--")
                axis.annotate(
                    label,
                    xy=(0, value),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 3),
                    textcoords="offset points",
                    color=MUTED,
                    fontsize=ANNOTATION_SIZE,
                )
        axis.set_xlabel("inference steps")
        axis.set_ylabel(Y_LABELS[task])
        axis.set_title(TITLES[task], color=TEXT)
        if len(data[task]) > 1:
            axis.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.21))
    figure.suptitle(
        "Error against inference steps: mean over seeds 0–4, band = 95% t-interval",
        color=TEXT,
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    output = arguments.output or arguments.results / "g3_steps.png"
    plot(curves(arguments.results), output)
    print(output)


if __name__ == "__main__":
    main()
