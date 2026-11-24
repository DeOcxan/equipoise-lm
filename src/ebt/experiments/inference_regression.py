"""G3 and G4 of the regression task on its final checkpoints (method.md 19).

    python -u -m ebt.experiments.inference_g3_g4 --task regression --tier full --device cpu

Imports the shared code (`inference_g3_g4`) and the regression experiment only, so the code
fingerprint of its results changes with the regression code and the shared code, not with the
code of the other tasks.
"""

import dataclasses
import math
from pathlib import Path

import torch
from torch import Tensor

from ebt.checkpoints import load_checkpoint
from ebt.config import ThinkingConfig, Tier
from ebt.core.energy import EnergyModel
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import regression_g2
from ebt.inference.annealed import accept_lower_step
from ebt.tasks.regression import BRANCH_TOLERANCE, branch_distance, initial_candidates
from ebt.tasks.regression import make_data as make_regression_data

EXPERIMENT = "inference_regression"
C3_DIVISORS = (1, 2, 4, 8, 16)
"""The (c3) inference experiment of G3: the trained α divided by these values."""
C3_MODEL = "ebt_regularized"
G3_MODELS = regression_g2.EBT_MODELS
"""Models whose inference is the thinking loop, so that a step count is defined."""
G4_MODELS = (*regression_g2.EBT_MODELS, "ibc", "ired")
"""Models with an energy of the final candidate."""
COVERAGE_MODELS = (*regression_g2.EBT_MODELS, "ired", "ibc", "mdn")
"""The models of the G4 coverage: N initial candidates (EBTs, IRED), N restarts of the
derivative-free optimizer (IBC) and N samples of the mixture (MDN)."""


def both_branch_coverage(x: Tensor, candidates: Tensor, sizes) -> list[dict]:
    """Fraction of rows whose first N candidates include one with |ŷ − √x| / √x ≤ 0.1 and one
    with |ŷ + √x| / √x ≤ 0.1 (r ≤ 0.1 on each branch, computed as `branch_distance` computes
    r); x (rows, 1), candidates (rows, M)."""
    root = x.reshape(-1, 1).sqrt()
    positive = (candidates - root).abs() / root <= BRANCH_TOLERANCE
    negative = (candidates + root).abs() / root <= BRANCH_TOLERANCE
    return [
        {
            "n": n,
            "coverage": (positive[:, :n].any(dim=1) & negative[:, :n].any(dim=1))
            .double()
            .mean()
            .item(),
        }
        for n in sizes
    ]


def accept_lower(
    model: EnergyModel, loop: ThinkingLoop, context: Tensor | None, candidate: Tensor
) -> Tensor:
    """The accept-only-lower step of method.md section 16.8, without a clamp."""
    return accept_lower_step(model, loop, context, candidate, math.inf)[0]


def fixed_step_loop(step_size: float, device: torch.device) -> ThinkingLoop:
    """A thinking loop with a fixed α, no jitter and no energy scaling (regression, 16.6)."""
    return ThinkingLoop(ThinkingConfig(step_size=step_size)).to(device)


def mixture_samples(mixture, generator: torch.Generator) -> Tensor:
    """y ~ Σ α_i N(μ_i, σ_i²) per row, shape (B, 1): a kernel drawn with probability α_i,
    then a Gaussian draw from it."""
    index = torch.multinomial(mixture.log_weights.exp(), 1, generator=generator)
    mean = mixture.means.gather(-1, index)
    scale = mixture.log_scales.exp().gather(-1, index)
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)
    return mean + scale * noise


def branch_curve(model, loop, test, init_std, seed, device, step=core.plain_step) -> dict:
    """Branch distance and on-branch fraction per step count of STEP_GRID, from the initial
    candidates of the g3_initial seed."""
    x = test.context.to(device)
    generator = torch.Generator().manual_seed(core.evaluation_seeds(seed)["g3_initial"])
    initial = initial_candidates(test, init_std, generator).to(device)
    distance, on_branch = {}, {}

    def record(k, candidate):
        r = branch_distance(x.cpu(), candidate.cpu())
        distance[k] = r.mean().item()
        on_branch[k] = (r <= BRANCH_TOLERANCE).double().mean().item()

    undefined = core.think(model, loop, x, initial, core.STEP_GRID, record, step)
    return {
        "branch_distance": core.curve(distance, undefined),
        "on_branch": core.curve(on_branch, undefined),
        "undefined_from_step": undefined,
    }


def regression_g3(setup, test, init_std: float, seed: int, device) -> dict:
    """G3 curve of one EBT checkpoint: branch distance and on-branch fraction per step count,
    from the same initial candidates for every step count."""
    model, loop = setup.objective.model, setup.objective.loop
    return {
        "steps": list(core.STEP_GRID),
        **branch_curve(model, loop, test, init_std, seed, device),
    }


def regression_c3(setup, test, init_std: float, seed: int, device) -> list[dict]:
    """The (c3) inference experiment: α / d for d in C3_DIVISORS, plain gradient steps and
    accept-only-lower steps, every setting on STEP_GRID."""
    model, trained = setup.objective.model, setup.objective.loop
    alpha = trained.step_size.raw.clamp(min=1e-4).item()
    rows = []
    for divisor in C3_DIVISORS:
        for lower in (False, True):
            loop = fixed_step_loop(alpha / divisor, device)
            step = accept_lower if lower else core.plain_step
            row = branch_curve(model, loop, test, init_std, seed, device, step)
            rows.append(
                {
                    "divisor": divisor,
                    "step_size": alpha / divisor,
                    "accept_lower": lower,
                    "steps": list(core.STEP_GRID),
                    **row,
                }
            )
    return rows


def final_energy(energy: EnergyModel, context: Tensor, candidate: Tensor) -> Tensor:
    with torch.no_grad():
        return energy.energy(context, energy.encode_candidate(candidate)).reshape(-1)


def regression_candidates(
    name: str, setup, test, init_std: float, seed: int, device
) -> tuple[Tensor, Tensor | None]:
    """32 final candidates per test row (rows, 32) and their energies (None for the MDN), from
    the g4_candidates seed."""
    generator_device = device if name in ("ibc", "ired") else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(
        core.evaluation_seeds(seed)["g4_candidates"]
    )
    split = test.to(device)
    finals, energies = [], []
    count = max(core.CANDIDATE_GRID)
    if name == "mdn":
        with torch.no_grad():
            mixture = setup.trained(split.context)
        cpu_mixture = type(mixture)(*(t.cpu() for t in dataclasses.astuple(mixture)))
        for _ in range(count):
            finals.append(mixture_samples(cpu_mixture, generator).reshape(-1))
        return torch.stack(finals, dim=1), None
    if name in regression_g2.EBT_MODELS:
        model, loop = setup.objective.model, setup.objective.loop
        for _ in range(count):
            initial = initial_candidates(test, init_std, generator).to(device)
            trajectory = loop.infer(
                model, split.context, initial, regression_g2.INFERENCE, None, final_energy=True
            )
            finals.append(trajectory.candidates[-1].cpu().reshape(-1))
            energies.append(trajectory.final_energy.cpu().reshape(-1))
        return torch.stack(finals, dim=1), torch.stack(energies, dim=1)
    for _ in range(count):
        predictions = setup.predict(split, generator)
        final = predictions.final.to(device)
        finals.append(final.cpu().reshape(-1))
        energies.append(final_energy(setup.energy, split.context, final).cpu())
    return torch.stack(finals, dim=1), torch.stack(energies, dim=1)


def regression_g4(name: str, x: Tensor, candidates: Tensor, energies: Tensor | None) -> dict:
    """Coverage and, with energies, Spearman and best-of-N; every value undefined when an
    energy or error is not finite (`core.g4_non_finite`)."""
    sizes = core.CANDIDATE_GRID
    errors = branch_distance(x, candidates)
    bad = core.g4_non_finite(candidates, errors, energies)
    g4: dict = {"candidates": max(sizes), "non_finite_values": bad}
    if bad:
        g4["coverage"] = [{"n": n, "coverage": None} for n in sizes]
        if energies is not None:
            g4["spearman"], g4["spearman_pairs"] = None, len(errors)
            g4["best_of_n"] = core.undefined_best_of_n(sizes)
        return g4
    g4["coverage"] = both_branch_coverage(x, candidates, sizes)
    if energies is not None:
        g4["spearman"] = core.spearman(energies[:, 0], errors[:, 0])
        g4["spearman_pairs"] = len(errors)
        g4["best_of_n"] = core.best_of_n(errors, energies, sizes)
    return g4


def evaluate(path: Path, device: torch.device, data, rows: int | None) -> dict:
    saved = load_checkpoint(path)
    name, seed = saved["run"]["model"], saved["run"]["seed"]
    settings = regression_g2.TIERS[Tier(saved["run"]["tier"])]
    data = make_regression_data(settings.data, saved["run"]["config"]["data_seed"])
    test = data.test if rows is None else data.test.rows(slice(0, rows))
    setup = regression_g2.restore(path, device)
    result: dict = {"error": "branch_distance", "test_rows": len(test)}
    if name in G3_MODELS:
        result["g3"] = regression_g3(setup, test, data.init_std, seed, device)
        result["trained_thinking"] = saved["run"]["config"]["thinking"]
        result["trained_inference"] = saved["run"]["config"]["inference"]
    if name == C3_MODEL:
        result["c3"] = regression_c3(setup, test, data.init_std, seed, device)
    if name in COVERAGE_MODELS or name in G4_MODELS:
        try:
            candidates, energies = regression_candidates(
                name, setup, test, data.init_std, seed, device
            )
        except FloatingPointError as error:
            result["g4"] = {
                "candidates": max(core.CANDIDATE_GRID),
                "non_finite_values": None,
                "floating_point_error": str(error),
                "coverage": [{"n": n, "coverage": None} for n in core.CANDIDATE_GRID],
                **(
                    {"spearman": None, "best_of_n": core.undefined_best_of_n(core.CANDIDATE_GRID)}
                    if name in G4_MODELS
                    else {}
                ),
            }
        else:
            result["g4"] = regression_g4(name, test.context.cpu(), candidates, energies)
    return result


TASK = core.Task(
    name="regression",
    module=EXPERIMENT,
    experiment=regression_g2.EXPERIMENT,
    models=COVERAGE_MODELS,
    report_seeds=tuple(regression_g2.REPORT_SEEDS),
    metric="branch_distance",
    lower_is_better=True,
    results="results/regression",
    evaluate=evaluate,
    load_data=lambda tier, directory: None,
)
