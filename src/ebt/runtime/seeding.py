"""Seeding of every random source used by the experiments (docs/numerics.md)."""

import torch


def seed_everything(seed: int) -> None:
    """Seed torch's global random number generator on all devices.

    torch.manual_seed seeds the generators of every device. The experiments draw all other
    randomness from explicit generators created with `make_generator`; they use neither
    Python's `random` module nor NumPy's random generator.
    """
    torch.manual_seed(seed)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    """A generator on `device` seeded with `seed`, for one source of randomness."""
    return torch.Generator(device=device).manual_seed(seed)


RUN_SEED_OFFSET = 1_000_000
RUN_SEED_STRIDE = 1_000


def run_seed(source: int, seed: int) -> int:
    """Seed of random source number `source` of the run with seed `seed`.

    The value is 1 000 000 + 1000 · source + seed. For run seeds 0–999 the values of
    different sources and runs are distinct, and all lie above the fixed seeds of the
    experiments (data splits and fixed evaluation noise, all below 1 000 000).
    """
    if not 0 <= seed < RUN_SEED_STRIDE:
        raise ValueError(f"run seed {seed} outside [0, {RUN_SEED_STRIDE})")
    return RUN_SEED_OFFSET + RUN_SEED_STRIDE * source + seed
