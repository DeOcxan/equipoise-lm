"""Explicit device selection (docs/numerics.md)."""

import torch

DEVICES = ("cpu", "mps")


def select_device(name: str) -> torch.device:
    """The requested device. There is no automatic choice and no fallback.

    Raises ValueError for an unknown name and RuntimeError when MPS is requested but not
    available, so that a run never silently changes device.
    """
    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {name!r}")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the MPS backend is not available on this machine")
    return torch.device(name)
