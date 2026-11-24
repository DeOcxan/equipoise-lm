"""The MNIST report tool draws its table and plots from result files only."""

import os
import subprocess
import sys
from pathlib import Path

import torch

from ebt.config import Tier
from ebt.experiments import mnist_denoising
from ebt.results import write_result
from test_mnist_training import DATA

ROOT = Path(__file__).resolve().parents[1]


def test_report_from_fast_tier_results(tmp_path):
    for model in mnist_denoising.MODELS:
        candidate = mnist_denoising.search_space(model, Tier.FAST)[0]
        for seed in (0, 1):
            result, _, _ = mnist_denoising.run(
                candidate, Tier.FAST, seed, torch.device("cpu"), True, DATA
            )
            write_result(tmp_path / f"{model}-seed{seed}.json", result)

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "tools.mnist_report", str(tmp_path)],
        check=True,
        env=environment,
        capture_output=True,
    )
    table = (tmp_path / "table.md").read_text()
    assert "EBT, S1, 2 inference steps" in table and "Noisy input" in table
    assert "2 seeds" in table and "seeds still falling" in table
    for name in ("examples.png", "validation.png"):
        assert (tmp_path / name).stat().st_size > 10_000
