"""Table and plots of question G2, generated from saved result files only.

usage: python -m tools.regression_g2_report RESULTS_DIR

Reads the per-seed result files of RESULTS_DIR (results/regression/g2/full for v1.0).

Outputs in RESULTS_DIR: table.md, predictions.png, energy_<model>.png for every energy model,
energy_ired_k9.png for the smoothest IRED landscape and density_mdn.png.
"""

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ebt.experiments.regression_g2 import MODELS
from ebt.results import RunResult, read_result
from ebt.statistics import mean_interval
from tools.palette import COLORS as TASK_COLORS
from tools.palette import GRID, MUTED, SURFACE, TEXT

LABELS = {
    "regressor": "(a) regressor, MSE",
    "mdn": "(a2) mixture density network, NLL",
    "ebt_mse": "(b) EBT, MSE",
    "ebt_contrastive": "(c1) EBT, contrastive",
    "ebt_combined": "(c2) EBT, MSE + contrastive",
    "ebt_regularized": "(c3) EBT, contrastive + energy penalty",
    "ibc": "(c4) IBC energy, InfoNCE, DFO inference",
    "ired": "(c5) IRED energy, 10 landscapes",
}
COLORS = TASK_COLORS["regression"]
DATA_COLOR = MUTED
BRANCH_COLOR = "#eb6834"
ENERGY_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLUMNS = [
    ("branch_distance", "mean r"),
    ("on_branch", "fraction r ≤ 0.1"),
    ("positive_branch", "fraction ŷ > 0"),
    ("initial_sign_agreement", "P(sign ŷ = sign ŷ₀)"),
    ("mean_move", "mean abs(ŷ − ŷ₀)"),
    ("mse", "test MSE"),
]
MSE_FLOOR = 2.125


def load(directory: Path) -> dict[str, list[RunResult]]:
    runs = {model: [] for model in MODELS}
    for path in sorted(directory.glob("*-seed*.json")):
        result = read_result(path)
        runs[result.model].append(result)
    return {
        model: sorted(results, key=lambda r: r.seed) for model, results in runs.items() if results
    }


def table(runs: dict[str, list[RunResult]]) -> str:
    """Markdown table: mean over seeds ± half-width of the 95% t-interval."""
    seeds = sorted({len(results) for results in runs.values()})
    header = "| Model | " + " | ".join(label for _, label in COLUMNS) + " |"
    lines = [
        f"Test split, mean over {', '.join(map(str, seeds))} seeds ± half-width of the 95% "
        f"t-interval. Mean-predictor MSE floor: {MSE_FLOOR} (method.md section 16.5).",
        "",
        header,
        "|" + "---|" * (len(COLUMNS) + 1),
    ]
    for model, results in runs.items():
        cells = []
        for key, _ in COLUMNS:
            values = [r.metrics["test"].get(key) for r in results]
            if any(value is None or math.isnan(value) for value in values):
                cells.append("n/a")
                continue
            mean, half = mean_interval(values)
            cells.append(f"{mean:.3f} ± {half:.3f}")
        lines.append(f"| {LABELS[model]} | " + " | ".join(cells) + " |")
    first = next(iter(runs.values()))[0]
    lines += [
        "",
        f"Mean of y² on the test split: {first.metrics['test']['mean_predictor_mse']:.4f}.",
    ]
    lines += diagnostic(runs)
    lines += readouts(runs)
    return "\n".join(lines) + "\n"


def diagnostic(runs: dict[str, list[RunResult]]) -> list[str]:
    """Mean |∂E/∂ŷ| at the test targets and at ŷ = 0, for the seed-0 runs that record it."""
    rows = [
        (model, results[0].artifacts["energy_gradient"])
        for model, results in runs.items()
        if "energy_gradient" in results[0].artifacts
    ]
    if not rows:
        return []
    lines = [
        "",
        "Energy gradient, seed 0, test split: mean abs(∂E/∂ŷ) at the targets (on the branches)",
        "and at ŷ = 0, and α times each, the size of one thinking step at those points.",
        "",
        "| Model | α | on branches | at ŷ = 0 | α · on branches | α · at ŷ = 0 |",
        "|---|---|---|---|---|---|",
    ]
    for model, values in rows:
        alpha = values["step_size"]
        on, zero = values["on_branches"], values["at_zero"]
        steps = "n/a | n/a" if alpha is None else f"{alpha * on:.3f} | {alpha * zero:.3f}"
        alpha_text = "n/a" if alpha is None else f"{alpha:.4g}"
        lines.append(f"| {LABELS[model]} | {alpha_text} | {on:.3f} | {zero:.3f} | {steps} |")
    lines += (
        [
            "",
            "(c4) takes no gradient steps. (c5): gradient on landscape k = 0 in the original",
            "units; α is σ_y² λ_0, the displacement in original units per unit gradient of its",
            "first descent step (method.md section 16.8).",
        ]
        if any(model in ("ibc", "ired") for model, _ in rows)
        else []
    )
    return lines


def readouts(runs: dict[str, list[RunResult]]) -> list[str]:
    """The mixture density network's other readouts and its negative log-likelihood."""
    if "mdn" not in runs:
        return []
    names = [
        ("test", "most probable branch, eq. 49 (table above)"),
        ("test_most_probable_value", "most probable value, eq. 48"),
        ("test_sampled_centre", "centre of a kernel drawn with probability α_i"),
    ]
    lines = [
        "",
        "(a2) readouts, test split:",
        "",
        "| Readout | mean r | fraction r ≤ 0.1 | fraction ŷ > 0 | test MSE |",
        "|---|---|---|---|---|",
    ]
    for key, label in names:
        cells = []
        for metric in ("branch_distance", "on_branch", "positive_branch", "mse"):
            mean, half = mean_interval([r.metrics[key][metric] for r in runs["mdn"]])
            cells.append(f"{mean:.3f} ± {half:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    mean, half = mean_interval([r.metrics["test_nll"]["nll"] for r in runs["mdn"]])
    lines += ["", f"(a2) test negative log-likelihood: {mean:.3f} ± {half:.3f}."]
    return lines


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot_predictions(runs: dict[str, list[RunResult]], path: Path) -> None:
    """Test data and the seed-0 predictions of each model, one panel per model.

    Each panel has its own y-axis, because a diverging model's predictions would otherwise
    flatten the other panels.
    """
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor=SURFACE)
    axes.flat[-1].set_visible(False)
    for axis, model in zip(axes.flat, MODELS, strict=False):
        style(axis)
        if model not in runs:
            axis.set_visible(False)
            continue
        points = runs[model][0].artifacts["predictions"]
        axis.scatter(points["x"], points["y"], s=8, color=DATA_COLOR, alpha=0.35, label="test data")
        axis.scatter(points["x"], points["final"], s=8, color=COLORS[model], label="prediction")
        axis.set_title(LABELS[model], color=TEXT, fontsize=11)
        axis.legend(loc="upper left", frameon=False, fontsize=9)
    for axis in axes.flat:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.suptitle("Test data and predictions (seed 0)", color=TEXT)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_landscape(grid: dict, values: np.ndarray, title: str, quantity: str, path: Path) -> None:
    """`values` on the (x, ŷ) grid minus its minimum over ŷ at each x, with the branches ±√x."""
    x, y = np.asarray(grid["x"]), np.asarray(grid["y"])
    relative = values - values.min(axis=1, keepdims=True)
    ceiling = np.quantile(relative, 0.95)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("energy", ENERGY_RAMP)

    figure, axis = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    image = axis.pcolormesh(x, y, relative.T, cmap=colormap, vmin=0, vmax=ceiling, shading="auto")
    for sign in (1, -1):
        axis.plot(x, sign * np.sqrt(x), color=BRANCH_COLOR, linewidth=2, linestyle="--")
    axis.plot([], [], color=BRANCH_COLOR, linewidth=2, linestyle="--", label="branches ±√x")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    axis.set_xlabel("x")
    axis.set_ylabel("candidate ŷ")
    axis.set_title(textwrap.fill(title, 60), color=TEXT, fontsize=10)
    bar = figure.colorbar(image, ax=axis)
    bar.set_label(f"relative {quantity}, clipped at the 95th percentile ({ceiling:.3g})")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_energy(result: RunResult, path: Path) -> None:
    """E(x, ŷ) minus its minimum over ŷ at each x (landscape k = 0 for IRED)."""
    grid = result.artifacts["energy_grid"]
    where = " on landscape k = 0" if result.model == "ired" else ""
    title = f"{LABELS[result.model]}: E(x, ŷ){where} − min over ŷ (seed 0)"
    plot_landscape(grid, np.asarray(grid["energy"]), title, "energy", path)


def plot_smoothest(result: RunResult, path: Path) -> None:
    grid = result.artifacts["energy_grid_smoothest"]
    title = f"{LABELS[result.model]}: E(x, ŷ) on landscape k = {grid['landscape']} (seed 0)"
    plot_landscape(grid, np.asarray(grid["energy"]), title, "energy", path)


def plot_density(result: RunResult, path: Path) -> None:
    grid = result.artifacts["density_grid"]
    title = f"{LABELS[result.model]}: −ln p(ŷ | x) − min over ŷ (seed 0)"
    values = np.asarray(grid["negative_log_density"])
    plot_landscape(grid, values, title, "negative log-density", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    runs = load(directory)
    text = table(runs)
    (directory / "table.md").write_text(text)
    print(text)
    plot_predictions(runs, directory / "predictions.png")
    for model, results in runs.items():
        found = results[0].artifacts
        if "energy_grid" in found:
            plot_energy(results[0], directory / f"energy_{model}.png")
        if "energy_grid_smoothest" in found:
            landscape = found["energy_grid_smoothest"]["landscape"]
            plot_smoothest(results[0], directory / f"energy_{model}_k{landscape}.png")
        if "density_grid" in found:
            plot_density(results[0], directory / f"density_{model}.png")


if __name__ == "__main__":
    main()
