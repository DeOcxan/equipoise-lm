"""Trained parameters of a final run, saved for inference experiments on the trained models.

A checkpoint holds the state_dict of the run's trained module (for an EBT the energy model and
the thinking loop, whose state includes the step size α), the code fingerprint, and the run's
model, tier, seed and config. The experiment that wrote it rebuilds the module from the config
and loads the state (`restore` of each experiment module).
"""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ebt.results import RunResult

REPOSITORY = Path(__file__).resolve().parents[2]
"""The repository root (src/ebt/checkpoints.py is two levels below it)."""


def checkpoint_directory(experiment: str, tier: str) -> Path:
    """REPOSITORY/checkpoints/<experiment>/<tier>, independent of the working directory."""
    return REPOSITORY / "checkpoints" / experiment / tier


def checkpoint_path(directory: Path, model: str, seed: int) -> Path:
    return directory / f"{model}-seed{seed}.pt"


def recorded_path(path: Path) -> str:
    """`path` as written into a result file: relative to the repository root if inside it."""
    path = path.resolve()
    return (path.relative_to(REPOSITORY) if path.is_relative_to(REPOSITORY) else path).as_posix()


def save_checkpoint(path: Path, module: nn.Module, fingerprint: str, result: RunResult) -> None:
    """Write the parameters and buffers of `module` on the CPU, the fingerprint, and the model,
    tier, seed and config of `result`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in module.state_dict().items()}
    run = {key: getattr(result, key) for key in ("experiment", "model", "tier", "seed", "config")}
    run = json.loads(json.dumps(run))  # plain values only, e.g. enums as their strings
    torch.save({"state_dict": state, "code_fingerprint": fingerprint, "run": run}, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """The saved dictionary: state_dict, code_fingerprint and run (tensors and plain values;
    loading runs no stored code)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_fingerprint(path: Path) -> str | None:
    """The code fingerprint stored in the checkpoint at `path`, or None if there is no file."""
    return load_checkpoint(path)["code_fingerprint"] if path.exists() else None
