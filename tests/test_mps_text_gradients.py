"""Float32-rounding decision for the MPS text-gradient test (tools/mps_text_gradients.py)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.mps_text_gradients import FACTOR, decide, round_up

ROOT = Path(__file__).resolve().parents[1]


def row(layout, cpu_abs=1e-5, cpu_norm=1e-7, mps_abs=2e-5, mps_norm=2e-7, difference=2.5e-5):
    def errors(absolute, normwise):
        return {"max_abs": absolute, "max_rel": 1e-6, "max_normwise_rel": normwise}

    return {
        "layout": layout,
        "cpu_float32": errors(cpu_abs, cpu_norm),
        "mps_float32": errors(mps_abs, mps_norm),
        "mps_vs_cpu_float32": errors(difference, 1e-7),
        "reference_max_abs": 3.0,
    }


def test_the_factor_is_2_5():
    assert FACTOR == 2.5


def test_within_the_factor_in_both_layouts_is_float32_rounding():
    decision = decide([row("full"), row("split", cpu_abs=1.17e-5)])
    assert decision.rounding and decision.within == {"full": True, "split": True}
    # (2.5 + 1) · 1.17e-5 = 4.095e-5 is above 1.5 · 2.5e-5 = 3.75e-5; rounded up, 4.1e-5.
    assert decision.atol == pytest.approx(4.1e-5)
    assert decision.lines[-1] == "DECISION: float32 rounding; atol = 4.1e-05"


def test_the_factor_is_inclusive():
    decision = decide([row("full", mps_abs=2.5e-5, mps_norm=2.5e-7), row("split")])
    assert decision.rounding


@pytest.mark.parametrize(
    "changed",
    [{"mps_abs": 2.6e-5}, {"mps_norm": 2.6e-7}],
    ids=["absolute error", "normwise relative error"],
)
@pytest.mark.parametrize("layout", ["full", "split"])
def test_each_criterion_in_each_layout_can_fail_the_decision(changed, layout):
    rows = [row(name, **(changed if name == layout else {})) for name in ("full", "split")]
    decision = decide(rows)
    assert not decision.rounding and decision.atol is None
    assert decision.within[layout] is False
    assert decision.lines[-1].startswith("DECISION: the MPS error is clearly larger")


def test_the_elementwise_relative_error_is_reported_but_does_not_decide():
    rows = [row("full"), row("split")]
    rows[0]["mps_float32"]["max_rel"] = 1.0
    assert decide(rows).rounding


def test_atol_is_at_least_one_and_a_half_times_the_measured_difference():
    # (2.5 + 1) · 1e-5 = 3.5e-5 is below 1.5 · 4.1e-5 = 6.15e-5, rounded up to 6.2e-5.
    decision = decide([row("full", difference=4.1e-5), row("split")])
    assert decision.atol == pytest.approx(6.2e-5)


def test_atol_within_float32_default_keeps_default_tolerances():
    """A derived atol within the float32 default keeps the default tolerances."""
    tiny = {"cpu_abs": 2e-6, "mps_abs": 3e-6, "difference": 3e-6}
    decision = decide([row("full", **tiny), row("split", **tiny)])
    # (2.5 + 1) · 2e-6 = 7e-6 and 1.5 · 3e-6 = 4.5e-6: atol 7e-6 ≤ 1e-5.
    assert decision.rounding and decision.atol is None
    assert "keeps its default tolerances" in decision.lines[-1]


def test_round_up_keeps_two_significant_digits():
    assert round_up(4.11e-5) == pytest.approx(4.2e-5)
    assert round_up(4.2e-5) == pytest.approx(4.2e-5)
    assert round_up(1.0001e-4) == pytest.approx(1.1e-4)


def test_both_layouts_are_required():
    with pytest.raises(ValueError):
        decide([row("full")])


def test_the_command_line_prints_the_decision_and_exits_by_it(tmp_path):
    def run(rows):
        path = tmp_path / "diagnosis.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return subprocess.run(
            [sys.executable, "-m", "tools.mps_text_gradients", "decide", str(path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

    passed = run([row("full"), row("split")])
    assert passed.returncode == 0 and "DECISION: float32 rounding" in passed.stdout
    failed = run([row("full", mps_abs=1e-3), row("split")])
    assert failed.returncode == 1 and "clearly larger" in failed.stdout
