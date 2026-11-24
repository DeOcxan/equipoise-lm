"""Result files: every schema field survives a write and read (questions.md)."""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from ebt.results import (
    FIELDS,
    SCHEMA_VERSION,
    RunResult,
    environment,
    read_json,
    read_result,
    write_json,
    write_result,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def strict_loads(text: str):
    """json.loads that rejects the non-standard tokens NaN, Infinity and -Infinity."""

    def reject(name):
        raise ValueError(f"non-standard JSON token {name}")

    return json.loads(text, parse_constant=reject)


def example() -> RunResult:
    return RunResult(
        experiment="regression_g2",
        task="regression",
        model="ebt_mse",
        tier="fast",
        seed=0,
        device="cpu",
        dtype="float32",
        environment=environment(),
        config={"thinking": {"mode": "s2", "step_size": 0.3}},
        parameters={"trainable": 4352},
        metrics={"test": {"mse": 2.1, "branch_distance": 0.9}},
        cost={"train_total_example_passes": {"forward": 10, "backward": 10, "second_order": 2}},
        history={"steps": [1, 2], "train_loss": [3.0, 2.5]},
        artifacts={"predictions": {"x": [1.0], "final": [0.0]}},
    )


def test_write_then_read_returns_every_field(tmp_path):
    path = tmp_path / "run.json"
    write_result(path, example())
    assert read_result(path) == example()
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert set(payload) == {"schema_version", *FIELDS}
    assert set(FIELDS) == {field.name for field in dataclasses.fields(RunResult)}


def test_environment_names_the_versions():
    assert {"python", "torch", "numpy", "matplotlib", "ebt", "platform"} <= set(environment())


def test_missing_field_and_wrong_version_are_rejected(tmp_path):
    path = tmp_path / "run.json"
    write_result(path, example())
    payload = json.loads(path.read_text())
    del payload["metrics"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing fields"):
        read_result(path)
    payload["metrics"], payload["schema_version"] = {}, SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema version"):
        read_result(path)


def test_undefined_metric_is_written_as_null_and_read_back_as_nan(tmp_path):
    path = tmp_path / "run.json"
    result = example()
    result.metrics["test"]["positive_branch_on_branch"] = math.nan
    result.artifacts["values"] = [1.0, math.nan]
    write_result(path, result)
    payload = strict_loads(path.read_text())
    assert payload["metrics"]["test"]["positive_branch_on_branch"] is None
    assert payload["artifacts"]["values"] == [1.0, None]
    assert math.isnan(read_result(path).metrics["test"]["positive_branch_on_branch"])


def test_infinity_is_rejected_and_non_standard_tokens_are_not_read(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "a.json", {"value": math.inf})
    path = tmp_path / "b.json"
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        read_json(path)


def test_every_result_file_in_the_repository_is_strict_json():
    paths = sorted(RESULTS.rglob("*.json"))
    assert paths
    for path in paths:
        strict_loads(path.read_text())
