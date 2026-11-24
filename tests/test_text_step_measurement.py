"""Text step measurement (tools/text_step_measurement.py) and its use by tools/text_budget.py."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import text_budget
from tools.text_budget import DoesNotFit, decide, key, record_key, run_measurement

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
TINY = ["--dim=64", "--depth=1", "--seq=8", "--batch=2", "--layout=split"]
SMALL = ["--device=cpu", "--warmup=0", "--repeats=1", "--validation-characters=65"]

EBT_FIELDS = {
    "dim",
    "depth",
    "seq",
    "batch",
    "model",
    "layout",
    "parameters",
    "step_k2_s",
    "step_k3_s",
    "step_k4_s",
    "step_mean_s",
    "peak_driver_k4_gib",
    "validation_eval_s",
    "generation_s_per_character",
}
BASELINE_FIELDS = {
    "dim",
    "depth",
    "seq",
    "batch",
    "model",
    "layout",
    "parameters",
    "step_s",
    "peak_driver_gib",
    "validation_eval_s",
}


def measure(model):
    command = [
        sys.executable,
        "-m",
        "tools.text_step_measurement",
        *TINY,
        *SMALL,
        f"--model={model}",
    ]
    done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=ENV, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_tiny_cpu_run_gives_every_field_the_rule_reads():
    ebt, base = measure("ebt"), measure("transformer")
    assert ebt.keys() >= EBT_FIELDS and base.keys() >= BASELINE_FIELDS
    for record in (ebt, base):
        assert record["memory_source"] == "process peak resident set size"
        assert all(record[f] > 0 for f in record if f.endswith(("_s", "_gib", "_character")))
    assert ebt["step_mean_s"] == pytest.approx(
        (ebt["step_k2_s"] + ebt["step_k3_s"] + ebt["step_k4_s"]) / 3
    )
    # The rule and the section text accept the records.
    records = {record_key(ebt): ebt, record_key(base): base}
    decision = decide(records, {}, 1e6)
    assert decision.missing["layout"] == "split" and decision.missing["dim"] == 384
    memory = {"date": "t", "free_percent": 50, "free_gib": 1e6, "total_gib": 2e6, "loadavg": "l"}
    assert "| 64 × 1 | 8 | 2 | EBT | split |" in text_budget.section(memory, decision, records, {})


FIELDS = {"dim": 64, "depth": 1, "seq": 8, "batch": 2, "model": "ebt", "layout": "split"}


class FinishedProcess:
    """Stands in for subprocess.Popen: a process that has already ended (no real signal)."""

    def __init__(self, returncode, stderr):
        self.returncode, self.stderr, self.pid = returncode, stderr, -1

    def __call__(self, *args, **kwargs):
        return self

    def communicate(self, timeout=None):
        return "", self.stderr


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(1, "RuntimeError: MPS backend out of memory"), (-9, "")],
    ids=["out-of-memory error", "killed by the system"],
)
def test_an_out_of_memory_process_is_recorded_as_out_of_memory(
    monkeypatch, tmp_path, returncode, stderr
):
    monkeypatch.setattr(text_budget.subprocess, "Popen", FinishedProcess(returncode, stderr))
    with pytest.raises(DoesNotFit) as failure:
        run_measurement(tmp_path, ["measure"], ENV, FIELDS)
    assert failure.value.reason == "out of memory"
    (line,) = (tmp_path / "oom.jsonl").read_text().splitlines()
    assert json.loads(line)["key"] == key(**FIELDS)
    assert not (tmp_path / "measurements.jsonl").exists()


def test_any_other_failure_stops(tmp_path):
    with pytest.raises(SystemExit) as stop:
        run_measurement(tmp_path, [sys.executable, "-c", "raise ValueError('x')"], ENV, FIELDS)
    assert stop.value.code == 1
    assert not (tmp_path / "oom.jsonl").exists()


FAKE = """
import argparse, json
parser = argparse.ArgumentParser()
for name in ("dim", "depth", "seq", "batch"):
    parser.add_argument(f"--{name}", type=int)
parser.add_argument("--layout"); parser.add_argument("--model")
a = parser.parse_args()
if a.dim == 384:
    raise SystemExit("RuntimeError: MPS backend out of memory")
record = vars(a) | {"parameters": 1000, "validation_eval_s": 0.1}
if a.model == "ebt":
    record |= {f"step_k{k}_s": 0.3 for k in (2, 3, 4)} | {"step_mean_s": 0.3,
               "peak_driver_k4_gib": 1.0, "generation_s_per_character": 0.01}
else:
    record |= {"step_s": 0.1, "peak_driver_gib": 0.5}
print(json.dumps(record))
"""


def test_the_command_line_measures_in_order_records_and_resumes(tmp_path):
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
        "--measure-command",
        f"{sys.executable} {fake}",
    ]

    first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=ENV, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    choice = json.loads((out / "choice.json").read_text())
    assert (choice["dim"], choice["sequence_length"], choice["batch_size"]) == (256, 256, 64)
    assert len((out / "oom.jsonl").read_text().splitlines()) == 4  # 384 x 6, split, 4 settings
    assert len((out / "measurements.jsonl").read_text().splitlines()) == 3
    assert "Chosen: width 256, depth 6" in (out / "section-18.8.md").read_text()

    before = {p.name: p.read_bytes() for p in out.iterdir() if p.suffix != ".log"}
    second = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=ENV, timeout=120)
    assert second.returncode == 0
    assert "start" not in second.stdout  # nothing measured again
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.suffix != ".log"} == before
