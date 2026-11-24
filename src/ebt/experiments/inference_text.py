"""G3 and G4 of the character-level language model on its final checkpoints (method.md 19).

    python -u -m ebt.experiments.inference_g3_g4 --task text --tier full --device mps

Imports the shared code (`inference_g3_g4`) and the text experiment only, so the code
fingerprint of its results changes with the text code and the shared code, not with the code
of the other tasks. Only the EBT is evaluated: the transformer has no thinking loop.
"""

import math
from pathlib import Path

import torch
from torch import Tensor

from ebt.checkpoints import load_checkpoint
from ebt.config import Tier
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import text_lm
from ebt.tasks.text import load_data

EXPERIMENT = "inference_text"


def position_bits(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy in bits of every position, (B, S)."""
    log_probabilities = logits.double().log_softmax(dim=-1)
    nats = -log_probabilities.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)
    return nats / math.log(2)


def evaluate(path: Path, device: torch.device, data, rows: int | None) -> dict:
    saved = load_checkpoint(path)
    seed = saved["run"]["seed"]
    settings = text_lm.TIERS[Tier(saved["run"]["tier"])]
    model, loop = text_lm.restore(path, device)
    test = data.test if rows is None else data.test.rows(slice(0, rows))
    vocabulary = model.config.vocabulary
    batch = settings.eval_batch
    seeds = core.evaluation_seeds(seed)

    # G3: BPC over every test position per step count, from N(0, 1) initial logits.
    generator = torch.Generator().manual_seed(seeds["g3_initial"])
    bits = {k: 0.0 for k in core.STEP_GRID}
    undefined = None
    for start in range(0, len(test), batch):
        part = test.rows(slice(start, start + batch))
        context, target = part.context.to(device), part.target

        def record(k, candidate, target=target):
            bits[k] += position_bits(candidate.cpu(), target).sum().item()

        count_rows, length = context.shape
        initial = text_lm.initial_logits(count_rows, length, vocabulary, generator).to(device)
        stop = core.think(model, loop, context, initial, core.STEP_GRID, record)
        if stop is not None:
            undefined = stop if undefined is None else min(undefined, stop)
    positions = test.target.numel()
    bpc = {k: bits[k] / positions for k in core.STEP_GRID}
    g3 = {
        "steps": list(core.STEP_GRID),
        "bpc": core.curve(bpc, undefined),
        "undefined_from_step": undefined,
    }

    # G4: 32 candidates per window, the trained inference (4 steps), per position.
    generator = torch.Generator().manual_seed(seeds["g4_candidates"])
    count = max(core.CANDIDATE_GRID)
    errors, energies = [], []
    g4: dict | None = None
    try:
        for start in range(0, len(test), batch):
            part = test.rows(slice(start, start + batch))
            context, target = part.context.to(device), part.target
            count_rows, length = context.shape
            row_errors, row_energies = [], []
            for _ in range(count):
                initial = text_lm.initial_logits(count_rows, length, vocabulary, generator)
                trajectory = loop.infer(
                    model, context, initial.to(device), text_lm.INFERENCE, None, final_energy=True
                )
                row_errors.append(position_bits(trajectory.candidates[-1].cpu(), target).flatten())
                row_energies.append(trajectory.final_energy.cpu().double().flatten())
            errors.append(torch.stack(row_errors, dim=1))
            energies.append(torch.stack(row_energies, dim=1))
    except FloatingPointError as error:
        g4 = {"non_finite_values": None, "floating_point_error": str(error)}
    else:
        errors, energies = torch.cat(errors), torch.cat(energies)
        bad = core.g4_non_finite(errors, energies)
        if bad:
            g4 = {"non_finite_values": bad}
    if g4 is not None:
        g4 = {
            "candidates": count,
            **g4,
            "spearman": None,
            "spearman_pairs": positions,
            "best_of_n": core.undefined_best_of_n(core.CANDIDATE_GRID),
        }
    else:
        g4 = {
            "candidates": count,
            "non_finite_values": 0,
            "spearman": core.spearman(energies[:, 0], errors[:, 0]),
            "spearman_pairs": len(errors),
            "best_of_n": core.best_of_n(errors, energies, core.CANDIDATE_GRID),
        }
    return {
        "error": "bits per character of each position",
        "test_rows": len(test),
        "g3": g3,
        "g4": g4,
        "trained_thinking": saved["run"]["config"]["thinking"],
        "trained_inference": saved["run"]["config"]["inference"],
    }


def load(tier: Tier, directory: Path | None):
    return load_data(directory or Path("data/text/raw"), text_lm.TIERS[tier].sequence_length)


TASK = core.Task(
    name="text",
    module=EXPERIMENT,
    experiment=text_lm.EXPERIMENT,
    models=("ebt",),
    report_seeds=tuple(text_lm.REPORT_SEEDS),
    metric="bpc",
    lower_is_better=True,
    results="results/text",
    evaluate=evaluate,
    load_data=load,
)
