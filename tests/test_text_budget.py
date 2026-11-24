"""Budget rule of method.md section 18.7 (tools/text_budget.py: decide)."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import text_budget
from tools.text_budget import (
    CONFIGS,
    LENGTHS,
    METAL_OUT_OF_MEMORY,
    OUT_OF_MEMORY,
    SAMPLE_STEPS,
    STAGE_BUDGET,
    TIMEOUT,
    Decision,
    DoesNotFit,
    decide,
    key,
    record_key,
    run_measurement,
    section,
)
from tools.text_step_measurement import GENERATIONS, REPEATS, STEP_COUNTS, WARMUP

FIXTURE = Path(__file__).parent / "data" / "text_budget_2026-11-20.jsonl"


def records_of(*records):
    return {record_key(r): r for r in records}


def ebt(dim, depth, seq, batch, layout="split", step=1.0, peak=1.0, **extra):
    return {
        "dim": dim,
        "depth": depth,
        "seq": seq,
        "batch": batch,
        "model": "ebt",
        "layout": layout,
        "parameters": 1000 + dim,
        "step_k2_s": step,
        "step_k3_s": step,
        "step_k4_s": step,
        "step_mean_s": step,
        "peak_driver_k4_gib": peak,
        "validation_eval_s": 0.1,
        "generation_s_per_character": 0.01,
        **extra,
    }


def baseline(dim, depth, seq, batch, step=0.1):
    return {
        "dim": dim,
        "depth": depth,
        "seq": seq,
        "batch": batch,
        "model": "transformer",
        "layout": "split",
        "parameters": 999 + dim,
        "step_s": step,
        "peak_driver_gib": 0.5,
        "validation_eval_s": 0.1,
    }


# EBT plus baseline step times (s) for which T = 5 000, 2 000 or 1 000 is the largest that fits
# 8 hours, and one for which none fits: the final runs need about 5 · T · (EBT + baseline step).
FITS_5000, FITS_2000, FITS_1000, FITS_NONE = 0.4, 1.9, 3.9, 6.9

ALL_SETTINGS = [(256, 64), (256, 32), (128, 64), (128, 32)]


def setting(dim, depth, seq, batch, step=FITS_5000, peak=1.0):
    return [
        ebt(dim, depth, seq, batch, "split", step, peak),
        ebt(dim, depth, seq, batch, "full", step * 2, peak * 2),
        baseline(dim, depth, seq, batch),
    ]


def run_rule(available, free_gib=10.0):
    """Apply the rule as the command line does: measure what it asks for, from `available`
    (a missing entry is out of memory). Returns the decision and the measurements in the order
    the rule asked for them."""
    records, failed, asked = {}, {}, []
    decision = decide(records, failed, free_gib)
    while decision.missing is not None:
        name = key(**decision.missing)
        asked.append(name)
        if name in available:
            records[name] = available[name]
        else:
            failed[name] = OUT_OF_MEMORY
        decision = decide(records, failed, free_gib)
    return decision, asked


def test_the_records_of_20_november_give_the_decision_of_that_date():
    """The records of method.md section 18.8 (measurements of 20 November 2026, item 3 as stated):
    64% of 16 GiB free, so a limit of 7.68 GiB. That sweep stopped at the out-of-memory error
    of 256 x 6, S 256, batch 64, full layout, before 384 x 6; that decision holds only if
    384 x 6 fitted at none of its settings, which is given here as out of memory."""
    records = records_of(*(json.loads(line) for line in FIXTURE.read_text().splitlines()))
    failed = {key(256, 6, 256, 64, "ebt", "full"): OUT_OF_MEMORY}

    assert decide(records, failed, 0.64 * 16).missing == {
        "dim": 384,
        "depth": 6,
        "seq": 256,
        "batch": 64,
        "model": "ebt",
        "layout": "split",
    }

    failed |= {key(384, 6, s, b, "ebt", "split"): OUT_OF_MEMORY for s, b in ALL_SETTINGS}
    decision = decide(records, failed, 0.64 * 16)
    assert decision.missing is None
    assert decision.limit_gib == pytest.approx(7.68)
    assert decision.choice == {
        "dim": 256,
        "depth": 6,
        "sequence_length": 256,
        "batch_size": 32,
        "layout": "split",
        "train_steps": 2000,
        "validate_every": 200,
        "search_hours": pytest.approx(5.08, abs=0.005),
        "final_hours": pytest.approx(6.39, abs=0.005),
        "ebt_parameters": 2789441,
        "baseline_parameters": 2789120,
    }
    assert decision.lines[4] == (
        "- 256 × 6, S 256, batch 64: faster layout split (k = 4: split 5.193 s, full out of "
        "memory); 11.73 GiB at k = 4, above 7.68 GiB."
    )
    assert decision.lines[5] == (
        "- 256 × 6, S 256, batch 32: faster layout split (k = 4: split 2.471 s, full 4.187 s); "
        "7.26 GiB at k = 4, within 7.68 GiB; mean EBT step 2.054 s, baseline 0.191 s; "
        "T = 1 000: search 2.58 h, finals 3.27 h; T = 2 000: search 5.08 h, finals 6.39 h; "
        "T = 5 000: search 12.56 h, finals 15.75 h."
    )


def test_four_settings_in_order_before_next_smaller_size():
    """384 x 6 is over the memory limit (7.5 GiB) at every setting; 256 x 6 fits at the first
    setting. The rule asks for exactly the measurements it needs, in its order."""
    available = {}
    for seq, batch in ALL_SETTINGS:
        available |= records_of(*setting(384, 6, seq, batch, peak=8.0))
    for dims in [(256, 6), (128, 4)]:
        for seq, batch in ALL_SETTINGS:
            available |= records_of(*setting(*dims, seq, batch))
    decision, asked = run_rule(available)

    expected = []
    for seq, batch in ALL_SETTINGS:
        expected += [
            key(384, 6, seq, batch, "ebt", "split"),
            key(384, 6, seq, batch, "ebt", "full"),
        ]
    expected += [
        key(256, 6, 256, 64, "ebt", "split"),
        key(256, 6, 256, 64, "ebt", "full"),
        key(256, 6, 256, 64, "transformer", "split"),
    ]
    assert asked == expected
    assert (decision.choice["dim"], decision.choice["sequence_length"]) == (256, 256)
    assert decision.choice["batch_size"] == 64
    assert all("above 7.50 GiB" in line for line in decision.lines[:4])


def test_larger_size_at_later_setting_beats_smaller_size():
    """A larger size at a later setting wins over a smaller size at the first."""
    available = records_of(*setting(384, 6, 128, 32), *setting(256, 6, 256, 64))
    for seq, batch in ALL_SETTINGS[:3]:
        available |= records_of(*setting(384, 6, seq, batch, peak=9.0))
    decision, _ = run_rule(available)
    choice = decision.choice
    assert (choice["dim"], choice["sequence_length"], choice["batch_size"]) == (384, 128, 32)


@pytest.mark.parametrize(
    "seq_batch", [(256, 64), (256, 32), (128, 64), (128, 32)], ids=lambda s: f"S{s[0]}B{s[1]}"
)
def test_the_first_setting_that_fits_is_chosen(seq_batch):
    available = {}
    for seq, batch in ALL_SETTINGS:
        fits = (seq, batch) == seq_batch
        available |= records_of(*setting(384, 6, seq, batch, peak=1.0 if fits else 9.0))
    decision, _ = run_rule(available)
    assert (decision.choice["sequence_length"], decision.choice["batch_size"]) == seq_batch


def test_the_memory_limit_is_75_percent_of_free_memory_and_inclusive():
    at_limit = records_of(*setting(384, 6, 256, 64, peak=7.5))
    assert run_rule(at_limit, free_gib=10.0)[0].choice["dim"] == 384
    above = records_of(*setting(384, 6, 256, 64, peak=7.5001), *setting(384, 6, 256, 32))
    decision, _ = run_rule(above, free_gib=10.0)
    assert decision.choice["batch_size"] == 32
    assert "7.50 GiB at k = 4, above 7.50 GiB" in decision.lines[0]


@pytest.mark.parametrize(
    ("step", "length"), [(FITS_5000, 5000), (FITS_2000, 2000), (FITS_1000, 1000)]
)
def test_the_longest_training_length_that_fits_both_stage_budgets_is_chosen(step, length):
    decision, _ = run_rule(records_of(*setting(384, 6, 256, 64, step=step - 0.1)))
    assert decision.choice["train_steps"] == length
    assert decision.choice["validate_every"] == length // 10
    assert decision.choice["final_hours"] <= 8 and decision.choice["search_hours"] <= 8


def test_a_setting_where_no_training_length_fits_is_passed_over():
    available = records_of(
        *setting(384, 6, 256, 64, step=FITS_NONE), *setting(384, 6, 256, 32, step=FITS_2000)
    )
    decision, _ = run_rule(available)
    assert decision.lines[0].endswith("T = 1 000 does not fit.")
    assert decision.choice["batch_size"] == 32


def test_layouts_the_faster_at_k4_and_out_of_memory_cases():
    split, full, base = setting(384, 6, 256, 64)
    split["step_k4_s"], full["step_k4_s"], full["peak_driver_k4_gib"] = 2.0, 1.0, 1.0
    decision, _ = run_rule(records_of(split, full, base))
    assert decision.choice["layout"] == "full"

    decision, asked = run_rule(records_of(split, base))  # full layout out of memory
    assert decision.choice["layout"] == "split"
    assert "full out of memory" in decision.lines[0]

    decision, asked = run_rule(records_of(full, base, *setting(384, 6, 256, 32)))
    assert key(384, 6, 256, 64, "ebt", "full") not in asked  # split out of memory: skipped
    assert (
        decision.lines[0] == "- 384 × 6, S 256, batch 64: the EBT ran out of memory (split layout)."
    )

    decision, _ = run_rule(records_of(split, full, *setting(384, 6, 256, 32)))
    assert decision.lines[0] == "- 384 × 6, S 256, batch 64: the baseline ran out of memory."
    assert decision.choice["batch_size"] == 32


def test_nothing_fits():
    available = {}
    for dims in [(384, 6), (256, 6), (128, 4)]:
        for seq, batch in ALL_SETTINGS:
            available |= records_of(*setting(*dims, seq, batch, step=FITS_NONE))
    decision, asked = run_rule(available)
    assert decision.choice is None and decision.missing is None
    assert len(decision.lines) == 12 and len(asked) == 36
    memory = {"date": "t", "free_percent": 50, "free_gib": 10.0, "total_gib": 20, "loadavg": "l"}
    text = section(memory, decision, available, {})
    assert "No size and setting meet items 2 and 4; the search does not start." in text


def test_a_missing_measurement_is_requested_before_any_decision():
    decision = decide({}, {}, 10.0)
    assert isinstance(decision, Decision) and decision.choice is None
    assert decision.missing == {
        "dim": 384,
        "depth": 6,
        "seq": 256,
        "batch": 64,
        "model": "ebt",
        "layout": "split",
    }


# Out of memory, the time limit and exit codes of the command line (tools/text_budget.py).

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
FIELDS = {"dim": 384, "depth": 6, "seq": 256, "batch": 64, "model": "ebt", "layout": "split"}


def python(script):
    return [sys.executable, "-c", script]


def failures(out):
    return [json.loads(line) for line in (out / "oom.jsonl").read_text().splitlines()]


class FinishedProcess:
    """Stands in for subprocess.Popen: a process that has already ended with `returncode` and
    `stderr`. No real process is started and no signal is sent."""

    def __init__(self, returncode, stderr):
        self.returncode, self.stderr, self.pid = returncode, stderr, -1

    def __call__(self, *args, **kwargs):
        return self

    def communicate(self, timeout=None):
        return "", self.stderr


def ended(monkeypatch, tmp_path, returncode, stderr=""):
    """The reason of the DoesNotFit that run_measurement raises for the ended process."""
    monkeypatch.setattr(text_budget.subprocess, "Popen", FinishedProcess(returncode, stderr))
    with pytest.raises(DoesNotFit) as failure:
        run_measurement(tmp_path, ["measure"], ENV, FIELDS)
    return failure.value.reason


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(137, ""), (-9, ""), (1, "RuntimeError: MPS backend out of memory")],
    ids=["exit 137", "signal 9", "out of memory message"],
)
def test_killed_or_out_of_memory_is_recorded_as_out_of_memory(
    monkeypatch, tmp_path, returncode, stderr
):
    assert ended(monkeypatch, tmp_path, returncode, stderr) == "out of memory"
    assert failures(tmp_path) == [
        {"key": key(**FIELDS), "exit": returncode, "reason": "out of memory"}
    ]


@pytest.mark.parametrize("text", METAL_OUT_OF_MEMORY)
@pytest.mark.parametrize("returncode", [-6, 134], ids=["signal 6", "exit 134"])
def test_a_metal_abort_on_a_memory_shortage_is_recorded_as_out_of_memory(
    monkeypatch, tmp_path, text, returncode
):
    stderr = f"-[MTLCommandBuffer] failed assertion: {text}"
    assert ended(monkeypatch, tmp_path, returncode, stderr) == "out of memory"
    assert failures(tmp_path)[0]["exit"] == returncode


@pytest.mark.parametrize("returncode", [-6, 134], ids=["signal 6", "exit 134"])
def test_an_abort_without_a_metal_memory_message_is_a_failure(monkeypatch, tmp_path, returncode):
    monkeypatch.setattr(
        text_budget.subprocess, "Popen", FinishedProcess(returncode, "failed assertion: other")
    )
    with pytest.raises(SystemExit) as stop:
        run_measurement(tmp_path, ["measure"], ENV, FIELDS)
    assert stop.value.code == 1
    assert not (tmp_path / "oom.jsonl").exists()


def test_a_measurement_over_the_time_limit_is_killed_with_its_process_group(tmp_path):
    """The measurement starts a child that would write a file after 2 s, then sleeps."""
    marker = tmp_path / "child-finished"
    script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen(['/bin/sh', '-c', 'sleep 2; touch {marker}']); time.sleep(60)"
    )
    start = time.monotonic()
    with pytest.raises(DoesNotFit) as failure:
        run_measurement(tmp_path, python(script), ENV, FIELDS, timeout=0.5)
    assert failure.value.reason == "timeout"
    assert time.monotonic() - start < 10
    assert failures(tmp_path)[0]["reason"] == "timeout"
    time.sleep(2.5)
    assert not marker.exists()


def test_over_the_time_limit_is_skipped_like_out_of_memory():
    split_key = key(384, 6, 256, 64, "ebt", "split")
    full_key = key(384, 6, 256, 64, "ebt", "full")
    base_key = key(384, 6, 256, 64, "transformer", "split")
    for reason in ("out of memory", "timeout"):
        decision = decide({}, {split_key: reason}, 10.0)
        assert decision.missing == {**FIELDS, "batch": 32}  # next setting, full not measured
    split, full, base = setting(384, 6, 256, 64)
    decision = decide(records_of(split, base), {full_key: "timeout"}, 10.0)
    assert decision.choice["layout"] == "split" and "full over time" in decision.lines[0]
    decision = decide(records_of(split, full), {base_key: "timeout"}, 10.0)
    assert decision.lines[0] == "- 384 × 6, S 256, batch 64: the baseline exceeded the time limit."
    assert decision.missing == {**FIELDS, "batch": 32}
    decision = decide({}, {split_key: "timeout"}, 10.0)
    assert decision.lines[0] == (
        "- 384 × 6, S 256, batch 64: the EBT exceeded the time limit (split layout)."
    )


FAKE = """
import argparse, json, sys, time
parser = argparse.ArgumentParser()
for name in ("dim", "depth", "seq", "batch"):
    parser.add_argument(f"--{name}", type=int)
parser.add_argument("--layout"); parser.add_argument("--model")
a = parser.parse_args()
if (a.dim, a.seq, a.batch) == (384, 256, 64):
    time.sleep(60)
if a.dim == 384:  # the exit code of a Metal abort, without a real abort
    sys.stderr.write("-[MTLCommandBuffer] kIOGPUCommandBufferCallbackErrorOutOfMemory")
    sys.exit(134)
record = vars(a) | {"parameters": 1000, "validation_eval_s": 0.1}
if a.model == "ebt":
    record |= {f"step_k{k}_s": 0.3 for k in (2, 3, 4)} | {"step_mean_s": 0.3,
               "peak_driver_k4_gib": 1.0, "generation_s_per_character": 0.01}
else:
    record |= {"step_s": 0.1, "peak_driver_gib": 0.5}
print(json.dumps(record))
"""


def test_the_command_line_records_a_timeout_and_a_metal_abort_and_goes_on(tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    out = tmp_path / "out"
    out.mkdir()
    memory = {"date": "t", "free_percent": 50, "total_gib": 16, "free_gib": 8.0, "loadavg": "l"}
    (out / "memory.json").write_text(json.dumps(memory))
    command = [
        sys.executable,
        "-m",
        "tools.text_budget",
        str(out),
        "--timeout",
        "1",
        "--measure-command",
        f"{sys.executable} {fake}",
    ]
    done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=ENV, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    reasons = [(r["key"], r["reason"]) for r in failures(out)]
    assert reasons[0] == ("384x6 S256 B64 ebt split", "timeout")
    assert [r for _, r in reasons[1:]] == ["out of memory"] * 3
    text = (out / "section-18.8.md").read_text()
    assert "Over the time limit of 1 s: 384x6 S256 B64 ebt split." in text
    assert "one that ran longer than 1 s is stopped" in text
    assert "the EBT exceeded the time limit (split layout)" in text
    assert json.loads((out / "choice.json").read_text())["dim"] == 256


@pytest.mark.parametrize(
    "arguments", [[], ["out", "--timeout", "soon"], ["out", "--unknown"]], ids=str
)
def test_a_usage_error_exits_with_1_never_with_2(tmp_path, arguments):
    command = [sys.executable, "-m", "tools.text_budget", *arguments]
    done = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, env=ENV, timeout=120
    )
    assert done.returncode == 1, done.stderr
    assert "error" in done.stderr


def test_the_memory_check_uses_the_faster_layout():
    """Full is faster than split and only full is within 75% of the free memory."""
    split, full, base = setting(384, 6, 256, 64)
    split |= {"step_k4_s": 2.0, "peak_driver_k4_gib": 9.0}
    full |= {"step_k4_s": 1.0, "step_mean_s": 1.0, "peak_driver_k4_gib": 6.0}
    decision = decide(records_of(split, full, base), {}, 10.0)
    assert (decision.choice["batch_size"], decision.choice["layout"]) == (64, "full")


def test_memory_exactly_at_75_percent_of_free_memory_fits():
    """Section 18.7 item 2: peak memory "at most 75%" of the free memory, so the limit fits."""
    split, full, base = setting(384, 6, 256, 64, peak=7.5)
    assert decide(records_of(split, full, base), {}, 10.0).choice["batch_size"] == 64
    split["peak_driver_k4_gib"] = math.nextafter(7.5, 8.0)
    assert decide(records_of(split, full, base), {}, 10.0).missing["batch"] == 32


def exactly_eight_hours(generation):
    """Records whose final runs at T = 1 000 project to 5 · (2 087.5 + 3 587.5 + 35)
    + 500 · `generation` (EBT) + 500 · `generation` (baseline, bounded by the EBT's) s:
    exactly 28 800 s = 8 h for generation 0.25 s."""
    split, full, base = setting(384, 6, 256, 64)
    split |= {
        "step_mean_s": 3.5,
        "validation_eval_s": 8.75,
        "generation_s_per_character": generation,
    }
    base |= {"step_s": 2.0, "validation_eval_s": 8.75}
    return records_of(split, full, base)


def test_a_projection_of_exactly_8_hours_fits():
    """Section 18.7 item 4: each stage must fit the budget of 8 hours, read as at most 8 h."""
    decision = decide(exactly_eight_hours(0.25), {}, 10.0)
    assert decision.choice["train_steps"] == 1000
    assert decision.choice["final_hours"] == 8.0
    over = decide(exactly_eight_hours(0.25 + 1e-9), {}, 10.0)
    assert over.choice is None and over.lines[0].endswith("T = 1 000 does not fit.")


def test_the_samples_are_one_generation_per_model():
    """text_lm.generate runs SAMPLE_STEPS characters for each model, both samples in one batch:
    the EBT's generation is counted once, and the baseline's once, at its own measured time if
    its record has one, otherwise at the EBT's."""
    split, _, base = setting(384, 6, 256, 64)

    def projected(ebt_character, baseline_character=None):
        ebt_record = split | {"generation_s_per_character": ebt_character}
        baseline_record = dict(base)
        if baseline_character is not None:
            baseline_record["generation_s_per_character"] = baseline_character
        return text_budget.projection_seconds(baseline_record, ebt_record, 1000)

    search, no_samples = projected(0.0, 0.0)
    assert projected(0.3)[1] - no_samples == pytest.approx(SAMPLE_STEPS * (0.3 + 0.3))
    assert projected(0.3, 0.1)[1] - no_samples == pytest.approx(SAMPLE_STEPS * (0.3 + 0.1))
    assert projected(0.3)[0] == search


def test_a_measurement_over_the_time_limit_could_not_have_fitted():
    """method.md section 18.7: a process over TIMEOUT spent more than TIMEOUT / 2 in its
    training steps and validation evaluation, or in its generations; either projects over 8 h
    at the smallest T."""
    steps = (WARMUP + REPEATS) * len(STEP_COUNTS)
    assert (WARMUP, REPEATS, steps, GENERATIONS, TIMEOUT) == (2, 5, 21, 10, 1800)
    for n in (WARMUP + REPEATS, steps):  # baseline, EBT
        # One run is T · m + 10 · v ≥ min(T / n, 10) · (n · m + v) > min(T / n, 10) · TIMEOUT / 2.
        assert min(min(LENGTHS) / n, 10) * TIMEOUT / 2 * CONFIGS > STAGE_BUDGET
    # The EBT's generation alone, without the baseline's.
    assert SAMPLE_STEPS * (TIMEOUT / 2 / GENERATIONS) > STAGE_BUDGET
