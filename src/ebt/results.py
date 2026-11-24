"""One JSON file per run, in a schema shared by every task.

Every reported number and plot is generated from these files (questions.md).
"""

import dataclasses
import json
import math
import platform
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

import ebt

SCHEMA_VERSION = 1


@dataclass
class RunResult:
    """Everything needed to report a run and to redraw its plots without the model."""

    experiment: str
    task: str
    model: str
    tier: str
    seed: int
    device: str
    dtype: str
    environment: dict[str, str]
    config: dict[str, Any]
    parameters: dict[str, int]
    metrics: dict[str, dict[str, float]]
    """Metrics per evaluation split, for example {"validation": {...}, "test": {...}}."""
    cost: dict[str, Any]
    history: dict[str, list]
    artifacts: dict[str, Any] = field(default_factory=dict)
    """Data for plots, such as predictions on a subset or an energy grid."""
    code: dict[str, Any] = field(default_factory=dict)
    """{"fingerprint": the code fingerprint (ebt.fingerprint) of the code that produced the run,
    "reused_stale_files": search entries and choices of other code that the run depended on}.
    Empty in files written before fingerprints were recorded."""


FIELDS = tuple(f.name for f in dataclasses.fields(RunResult))
REQUIRED = tuple(
    f.name
    for f in dataclasses.fields(RunResult)
    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
)


def environment() -> dict[str, str]:
    """Python, package versions and platform of the current process."""
    packages = {name: metadata.version(name) for name in ("numpy", "matplotlib")}
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ebt": ebt.__version__,
        "platform": platform.platform(),
        **packages,
    }


def _undefined_to_null(value: Any) -> Any:
    """NaN, an undefined value such as a fraction of an empty set, becomes None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _undefined_to_null(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_undefined_to_null(item) for item in value]
    return value


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON with sorted keys, so identical payloads give identical files.

    NaN is written as null. Any other value strict JSON cannot hold, such as an infinity,
    raises ValueError.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_undefined_to_null(payload), indent=1, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")


def read_json(path: Path) -> Any:
    """Read strict JSON; the non-standard tokens NaN and Infinity raise ValueError."""
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def write_result(path: Path, result: RunResult) -> None:
    """Write `result` with `write_json`."""
    write_json(path, {"schema_version": SCHEMA_VERSION, **dataclasses.asdict(result)})


def read_result(path: Path) -> RunResult:
    """Read a result file; raises ValueError for another schema version or a missing field.

    Fields with a default (artifacts, code) may be absent and then take their default.

    Metrics written as null, because they were undefined, are read back as NaN.
    """
    payload = read_json(path)
    version = payload.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema version {version}, expected {SCHEMA_VERSION}")
    missing = [name for name in REQUIRED if name not in payload]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    payload["metrics"] = {
        split: {key: math.nan if value is None else value for key, value in values.items()}
        for split, values in payload["metrics"].items()
    }
    return RunResult(**{name: payload[name] for name in FIELDS if name in payload})
