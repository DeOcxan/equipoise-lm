"""The text report tool draws its table, samples and plot from result files only."""

import os
import subprocess
import sys
from pathlib import Path

import torch

from ebt.config import Tier
from ebt.experiments import text_lm
from ebt.results import write_result
from test_text_training import DATA
from tools.text_report import load, training_passes

ROOT = Path(__file__).resolve().parents[1]


def test_report_from_fast_tier_results(tmp_path):
    for model in text_lm.MODELS:
        candidate = text_lm.search_space(model, Tier.FAST, tmp_path)[0]
        for seed in (0, 1):
            result, _, _ = text_lm.run(candidate, Tier.FAST, seed, torch.device("cpu"), True, DATA)
            write_result(tmp_path / f"{model}-seed{seed}.json", result)

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "tools.text_report", str(tmp_path)],
        check=True,
        env=environment,
        capture_output=True,
    )
    table = (tmp_path / "table.md").read_text()
    assert "EBT, S2, 4 inference steps" in table and "Causal transformer" in table
    assert "2 seeds" in table and "out-of-distribution" in table
    assert "Sample 2:" in (tmp_path / "samples.md").read_text()
    assert (tmp_path / "validation.png").stat().st_size > 10_000


def test_training_passes_are_the_mean_over_seeds_with_the_per_seed_range():
    """The recorded final runs: EBT forward and backward passes per fresh window 284 912,
    284 080, 287 280, 291 648 and 287 968 over 64 000 windows, second-order 95 984 on every
    seed; the transformer 1, 1 and 0."""
    runs = load(Path(__file__).parent.parent / "results" / "text" / "lm" / "full")
    ebt = "4.48715 (4.43875 to 4.55700)"
    assert training_passes(runs["ebt"]) == f"{ebt} / {ebt} / 1.49975"
    assert training_passes(runs["transformer"]) == "1 / 1 / 0"
