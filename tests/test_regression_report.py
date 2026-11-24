"""The G2 report tool draws its table and plots from result files only."""

import os
import subprocess
import sys
from pathlib import Path

import torch

from ebt.config import Tier
from ebt.experiments import regression_g2
from ebt.results import write_result

ROOT = Path(__file__).resolve().parents[1]


def test_report_from_fast_tier_results(tmp_path):
    for model in regression_g2.MODELS:
        candidate = regression_g2.search_space(model, Tier.FAST)[0]
        result, _, _ = regression_g2.run(
            candidate, Tier.FAST, 0, torch.device("cpu"), with_test=True
        )
        write_result(tmp_path / f"{model}-seed0.json", result)
        # A second seed for the interval: the same run relabelled, which the tool cannot tell
        # apart and which keeps the test fast.
        result.seed, result.artifacts = 1, {}
        write_result(tmp_path / f"{model}-seed1.json", result)

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "tools.regression_g2_report", str(tmp_path)],
        check=True,
        env=environment,
        capture_output=True,
    )
    table = (tmp_path / "table.md").read_text()
    assert "(c3) EBT, contrastive + energy penalty" in table and "2 seeds" in table
    assert "Energy gradient, seed 0" in table
    assert "(c5) IRED energy" in table and "(a2) readouts" in table
    expected = [
        "predictions.png",
        "energy_ebt_mse.png",
        "energy_ebt_contrastive.png",
        "energy_ebt_combined.png",
        "energy_ebt_regularized.png",
        "energy_ibc.png",
        "energy_ired.png",
        "energy_ired_k9.png",
        "density_mdn.png",
    ]
    for name in expected:
        assert (tmp_path / name).stat().st_size > 10_000
