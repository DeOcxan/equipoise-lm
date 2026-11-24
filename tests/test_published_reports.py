"""Every published report regenerates from the committed per-seed result files.

Each report is rebuilt with the report functions of tools/ and src/ as they are, written to
tmp_path and compared with the committed file: JSON number by number, tables character by
character. Only result files are read: no checkpoint, no data and no device. Figures (PNG) are
not compared and are not drawn.
"""

import json
import math
from pathlib import Path

import pytest

from ebt.experiments import inference_g3_g4 as core
from ebt.fingerprint import code_fingerprint
from ebt.results import read_json, write_json
from tools import g5_report, mnist_report, regression_g2_report, text_report

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "results"
TASKS = ("regression", "mnist", "text")
TIER = "full"


def assert_same(produced, committed, where: str = "") -> None:
    """Equal structure and values; floats equal to 12 significant digits."""
    if isinstance(committed, dict):
        assert isinstance(produced, dict), where
        assert sorted(produced) == sorted(committed), f"{where}: keys differ"
        for key in committed:
            assert_same(produced[key], committed[key], f"{where}.{key}")
    elif isinstance(committed, list):
        assert isinstance(produced, list), where
        assert len(produced) == len(committed), f"{where}: lengths differ"
        for i, (a, b) in enumerate(zip(produced, committed, strict=True)):
            assert_same(a, b, f"{where}[{i}]")
    elif isinstance(committed, float) and not isinstance(committed, bool):
        assert isinstance(produced, (int, float)), where
        assert math.isclose(produced, committed, rel_tol=1e-12, abs_tol=0.0), (
            f"{where}: {produced!r} != {committed!r}"
        )
    else:
        assert produced == committed, f"{where}: {produced!r} != {committed!r}"


def written(path: Path, payload) -> object:
    """The payload as it is read back from a file written by write_json."""
    write_json(path, payload)
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", TASKS)
def test_g3_g4_summary_and_table(name, tmp_path):
    task = core.load_task(name)
    output = RESULTS / name / "g3_g4" / TIER
    committed = read_json(output / "summary.json")
    summary = core.summarize(task, list(task.models), list(task.report_seeds), output)
    # The three fields execute() adds after summarize(); a full run reuses no stale file.
    summary["reused_stale_files"] = []
    summary["code_fingerprint"] = code_fingerprint(task.module)
    summary["fingerprint_module"] = task.module
    assert_same(written(tmp_path / "summary.json", summary), committed, f"{name} summary")
    (tmp_path / "table.md").write_text(core.table(task.name, summary))
    assert (tmp_path / "table.md").read_text() == (output / "table.md").read_text()


@pytest.mark.parametrize("name", TASKS)
def test_g5_report_and_table(name, tmp_path):
    directory = RESULTS / name / "g5" / TIER
    report = g5_report.task_report(RESULTS, name, TIER)
    assert_same(written(tmp_path / "g5.json", report), read_json(directory / "g5.json"), name)
    (tmp_path / "table.md").write_text(g5_report.table(report))
    assert (tmp_path / "table.md").read_text() == (directory / "table.md").read_text()


def search_of(directory: Path):
    path = directory / "search.json"
    return read_json(path) if path.exists() else None


@pytest.mark.parametrize(
    ("directory", "render"),
    [
        ("regression/g2", lambda d: regression_g2_report.table(regression_g2_report.load(d))),
        ("mnist/denoising", lambda d: mnist_report.table(mnist_report.load(d), search_of(d))),
        ("text/lm", lambda d: text_report.table(text_report.load(d), search_of(d))),
    ],
)
def test_training_table(directory, render, tmp_path):
    source = RESULTS / directory / TIER
    (tmp_path / "table.md").write_text(render(source))
    assert (tmp_path / "table.md").read_text() == (source / "table.md").read_text()
