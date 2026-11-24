"""G3 and G4 of the MNIST denoising task on its final checkpoints (method.md 19).

    python -u -m ebt.experiments.inference_g3_g4 --task mnist --tier full --device mps

Imports the shared code (`inference_g3_g4`) and the MNIST experiment only, so the code
fingerprint of its results changes with the MNIST code and the shared code, not with the code
of the other tasks.
"""

import math
from pathlib import Path

import torch
from torch import Tensor

from ebt.checkpoints import load_checkpoint
from ebt.config import Tier
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import mnist_denoising
from ebt.tasks.mnist import alpha_bar, load_images, make_data, to_pixels

EXPERIMENT = "inference_mnist"
PERTURBATION_STD = math.sqrt(1 - alpha_bar(mnist_denoising.NOISE_LEVEL))
"""σ of the perturbation of the MNIST G4 candidates: √(1 − ᾱ) = 0.3209 at noise level 0.1,
the scale of the corruption already in the noisy image; fixed, not tuned."""
PSNR_FIELDS = ("lowest_energy_psnr", "oracle_psnr")


def pixel_mse(prediction: Tensor, clean: Tensor) -> Tensor:
    """Per-image mean squared error in pixel units [0, 255] after the clamp of method.md 17.6;
    its mean over images is the MSE of the PSNR."""
    difference = to_pixels(prediction.cpu()).double() - to_pixels(clean.cpu()).double()
    return difference.square().flatten(1).mean(dim=1)


def psnr_of(mse: Tensor) -> float:
    return 10 * math.log10(255**2 / mse.mean().item())


def undefined_g4(count: int, pairs: int, bad: int | None, reason: str | None = None) -> dict:
    g4 = {
        "candidates": count,
        "perturbation_std": PERTURBATION_STD,
        "non_finite_values": bad,
        "spearman": None,
        "spearman_perturbed_first": None,
        "spearman_pairs": pairs,
        "standard_psnr": None,
        "perturbed_first_psnr": None,
        "best_of_n": core.undefined_best_of_n(core.CANDIDATE_GRID, PSNR_FIELDS),
    }
    if reason is not None:
        g4["floating_point_error"] = reason
    return g4


def evaluate(path: Path, device: torch.device, data, rows: int | None) -> dict:
    saved = load_checkpoint(path)
    seed = saved["run"]["seed"]
    settings = mnist_denoising.TIERS[Tier(saved["run"]["tier"])]
    model, loop = mnist_denoising.restore(path, device)
    clean = data.test.target if rows is None else data.test.target[:rows]
    noisy = mnist_denoising.fixed_noise(data.test.target, mnist_denoising.TEST_NOISE_SEED)
    noisy = noisy[: len(clean)]
    batch = settings.eval_batch
    inference = mnist_denoising.INFERENCE

    # G3: from the noisy image, as trained; one MSE over all pixels per step count.
    squared = {k: 0.0 for k in core.STEP_GRID}
    undefined = None
    for start in range(0, len(clean), batch):
        part = slice(start, start + batch)

        def record(k, candidate, part=part):
            squared[k] += pixel_mse(candidate, clean[part]).sum().item()

        stop = core.think(model, loop, None, noisy[part].to(device), core.STEP_GRID, record)
        if stop is not None:
            undefined = stop if undefined is None else min(undefined, stop)
    psnr = {k: 10 * math.log10(255**2 / (squared[k] / len(clean))) for k in core.STEP_GRID}
    g3 = {
        "steps": list(core.STEP_GRID),
        "psnr": core.curve(psnr, undefined),
        "undefined_from_step": undefined,
    }

    # G4: the standard output and 32 candidates from perturbed starts, as trained.
    generator = torch.Generator().manual_seed(core.evaluation_seeds(seed)["g4_candidates"])
    count = max(core.CANDIDATE_GRID)
    standard_error, standard_energy, errors, energies = [], [], [], []
    try:
        for start in range(0, len(clean), batch):
            part = slice(start, start + batch)
            start_image = noisy[part].to(device)
            trajectory = loop.infer(model, None, start_image, inference, None, final_energy=True)
            standard_error.append(pixel_mse(trajectory.candidates[-1], clean[part]))
            standard_energy.append(trajectory.final_energy.cpu())
            row_errors, row_energies = [], []
            for _ in range(count):
                perturbation = PERTURBATION_STD * torch.randn(
                    noisy[part].shape, generator=generator
                )
                initial = (noisy[part] + perturbation).to(device)
                trajectory = loop.infer(model, None, initial, inference, None, final_energy=True)
                row_errors.append(pixel_mse(trajectory.candidates[-1], clean[part]))
                row_energies.append(trajectory.final_energy.cpu())
            errors.append(torch.stack(row_errors, dim=1))
            energies.append(torch.stack(row_energies, dim=1))
    except FloatingPointError as error:
        g4 = undefined_g4(count, len(clean), None, str(error))
    else:
        standard_error, standard_energy = torch.cat(standard_error), torch.cat(standard_energy)
        errors, energies = torch.cat(errors), torch.cat(energies)
        bad = core.g4_non_finite(standard_error, standard_energy, errors, energies)
        g4 = undefined_g4(count, len(standard_error), bad) if bad else None
    if g4 is None:
        selection = core.best_of_n(errors, energies, core.CANDIDATE_GRID)
        for row in selection:
            n = row["n"]
            chosen = errors[:, :n].gather(1, energies[:, :n].argmin(dim=1, keepdim=True))
            row["lowest_energy_psnr"] = psnr_of(chosen)
            row["oracle_psnr"] = psnr_of(errors[:, :n].min(dim=1).values)
        g4 = {
            "candidates": count,
            "perturbation_std": PERTURBATION_STD,
            "non_finite_values": 0,
            "spearman": core.spearman(standard_energy, standard_error),
            "spearman_perturbed_first": core.spearman(energies[:, 0], errors[:, 0]),
            "spearman_pairs": len(standard_error),
            "standard_psnr": psnr_of(standard_error),
            "perturbed_first_psnr": psnr_of(errors[:, 0]),
            "best_of_n": selection,
        }
    return {
        "error": "per-image pixel MSE (PSNR for G3)",
        "test_rows": len(clean),
        "g3": g3,
        "g4": g4,
        "trained_thinking": saved["run"]["config"]["thinking"],
        "trained_inference": saved["run"]["config"]["inference"],
    }


def load(tier: Tier, directory: Path | None):
    images = load_images(directory or Path("data/mnist/raw"))
    return make_data(*images, mnist_denoising.VALIDATION_SIZE, mnist_denoising.DATA_SEED)


TASK = core.Task(
    name="mnist",
    module=EXPERIMENT,
    experiment=mnist_denoising.EXPERIMENT,
    models=("ebt",),
    report_seeds=tuple(mnist_denoising.REPORT_SEEDS),
    metric="psnr",
    lower_is_better=False,
    results="results/mnist",
    evaluate=evaluate,
    load_data=load,
)
