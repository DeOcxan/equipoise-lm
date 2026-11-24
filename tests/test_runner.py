"""Selection and file naming of the shared experiment runner (method.md section 16.6)."""

import math
import re
from dataclasses import asdict, dataclass

import pytest
import torch

from ebt.checkpoints import REPOSITORY, checkpoint_directory, recorded_path, save_checkpoint
from ebt.experiments import runner
from ebt.experiments.runner import (
    IncompleteSearchError,
    Provenance,
    StaleResultsError,
    candidate_name,
    choose,
    final,
    same_configuration,
)
from ebt.results import RunResult, read_json, write_json, write_result


def entries(*values):
    return [{"candidate": {"index": i}, "selection_value": value} for i, value in enumerate(values)]


def test_smallest_value_is_chosen_by_exact_comparison():
    """0.0906060716 and 0.0906061607 differ by 9e-8; the exact minimum wins."""
    chosen = choose(entries(0.0906061607, 0.0906060716, 0.1))
    assert chosen["candidate"]["index"] == 1


def test_first_in_grid_order_wins_an_exact_tie():
    assert choose(entries(0.5, 0.25, 0.25))["candidate"]["index"] == 1


def test_undefined_values_are_never_chosen_unless_all_are_undefined():
    assert choose(entries(None, math.nan, 3.0))["candidate"]["index"] == 2
    assert choose(entries(-math.inf, math.inf, 3.0))["candidate"]["index"] == 2
    assert choose(entries(None, math.nan))["candidate"]["index"] == 0


@dataclass(frozen=True)
class Example:
    model: str
    learning_rate: float
    step_size: float | None = None


def test_candidate_name_lists_fields_that_are_set_in_sorted_order():
    assert (
        candidate_name(Example("ebt", 1e-3, 30.0)) == "learning_rate=0.001_model=ebt_step_size=30.0"
    )
    assert candidate_name(Example("denoiser", 1e-3)) == "learning_rate=0.001_model=denoiser"


def test_same_configuration_ignores_fields_set_to_none():
    assert same_configuration({"a": 1, "b": None}, {"a": 1})
    assert not same_configuration({"a": 1}, {"a": 2})


def result_of(candidate, seed):
    return RunResult(
        experiment="e",
        task="t",
        model=candidate.model,
        tier="fast",
        seed=seed,
        device="cpu",
        dtype="float32",
        environment={},
        config={"candidate": asdict(candidate)},
        parameters={},
        metrics={"validation": {"value": float(candidate.learning_rate)}},
        cost={},
        history={},
    )


class Recorder:
    """A run function that records its calls and returns the configured selection values."""

    def __init__(self, values=None):
        self.calls = []
        self.values = values or {}

    def __call__(self, candidate, seed, with_test):
        self.calls.append((candidate, seed, with_test))
        value = self.values.get(candidate.learning_rate, candidate.learning_rate)
        if isinstance(value, Exception):
            raise value
        return result_of(candidate, seed), ("value", value), torch.nn.Linear(1, 1)


GRID = [Example("ebt", 3e-3), Example("ebt", 1e-3), Example("ebt", 3e-4)]
CURRENT = Provenance("current code")


def execute(path, run, provenance=CURRENT, stage="all"):
    checkpoints = path / "checkpoints"
    runner.execute(path, stage, ["ebt"], grid_of, Example, run, 100, (0,), provenance, checkpoints)


def finished_search(tmp_path):
    """A complete search on disk; returns search.json and the chosen configuration."""
    execute(tmp_path, Recorder(), stage="search")
    record = read_json(tmp_path / "search.json")
    return record, Example(**record["models"]["ebt"]["chosen"])


def grid_of(model):
    """The search space of every model: GRID."""
    return GRID


def run_final(tmp_path, record, run, seeds=(0, 1), provenance=CURRENT):
    final(tmp_path, record, ["ebt"], grid_of, Example, run, seeds, provenance, tmp_path / "ckpt")


def test_final_reruns_a_file_that_holds_another_configuration(tmp_path):
    record, chosen = finished_search(tmp_path)
    run_final(tmp_path, record, Recorder(), seeds=(1,))
    write_result(tmp_path / "ebt-seed0.json", result_of(Example("ebt", 1e-4, 30.0), 0))

    run = Recorder()
    run_final(tmp_path, record, run)

    assert run.calls == [(chosen, 0, True)]
    assert read_json(tmp_path / "ebt-seed0.json")["config"]["candidate"] == asdict(chosen)


def test_final_stops_when_a_search_configuration_is_missing(tmp_path):
    record, _ = finished_search(tmp_path)
    (tmp_path / "search" / "ebt" / f"{candidate_name(GRID[1])}.json").unlink()
    tried = record["models"]["ebt"]["tried"]
    tried[:] = [e for e in tried if e["candidate"] != asdict(GRID[1])]
    run = Recorder()
    with pytest.raises(IncompleteSearchError, match=re.escape(candidate_name(GRID[1]))):
        run_final(tmp_path, record, run)
    assert run.calls == []


def test_a_configuration_listed_only_in_search_json_counts_as_on_disk(tmp_path):
    """Searches written before the per-configuration files existed."""
    record, chosen = finished_search(tmp_path)
    for candidate in GRID:
        (tmp_path / "search" / "ebt" / f"{candidate_name(candidate)}.json").unlink()
    run = Recorder()
    run_final(tmp_path, record, run, seeds=(0,))
    assert run.calls == [(chosen, 0, True)]


def test_final_stops_when_recorded_choice_differs_from_fresh_one(tmp_path):
    """Final stops when the recorded choice differs from a fresh selection."""
    record, _ = finished_search(tmp_path)
    record["models"]["ebt"]["chosen"] = asdict(GRID[0])
    run = Recorder()
    with pytest.raises(IncompleteSearchError, match="configurations on disk give"):
        run_final(tmp_path, record, run)
    assert run.calls == []


@pytest.mark.parametrize("damage", ["no checkpoint", "stale result", "stale checkpoint"])
def test_final_run_skipped_only_with_current_result_and_checkpoint(tmp_path, damage):
    """A final run is skipped only with its result and checkpoint from current code."""
    record, chosen = finished_search(tmp_path)
    run_final(tmp_path, record, Recorder(), seeds=(0,))
    result_file, checkpoint = tmp_path / "ebt-seed0.json", tmp_path / "ckpt" / "ebt-seed0.pt"
    if damage == "no checkpoint":
        checkpoint.unlink()
    elif damage == "stale result":
        stored = read_json(result_file)
        stored["code"]["fingerprint"] = "other code"
        write_json(result_file, stored)
    else:
        save_checkpoint(checkpoint, torch.nn.Linear(1, 1), "other code", result_of(chosen, 0))

    run = Recorder()
    run_final(tmp_path, record, run, seeds=(0,))
    assert run.calls == [(chosen, 0, True)]
    assert read_json(result_file)["code"]["fingerprint"] == "current code"
    assert runner.checkpoint_fingerprint(checkpoint) == "current code"

    again = Recorder()
    run_final(tmp_path, record, again, seeds=(0,))
    assert again.calls == []


def test_the_checkpoint_directory_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = checkpoint_directory("text_lm", "full")
    assert directory == REPOSITORY / "checkpoints" / "text_lm" / "full"
    assert directory.is_absolute() and (REPOSITORY / "pyproject.toml").exists()
    assert recorded_path(directory / "ebt-seed0.pt") == "checkpoints/text_lm/full/ebt-seed0.pt"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, FloatingPointError("diverged")])
def test_a_non_finite_selection_value_is_recorded_as_null_and_ranks_last(tmp_path, bad):
    """The search goes on; the configuration is written with value null and the reason."""
    run = Recorder({3e-4: bad})
    execute(tmp_path, run)

    entry = read_json(tmp_path / "search" / "ebt" / "learning_rate=0.0003_model=ebt.json")
    assert entry["selection_value"] is None
    assert entry["selection_value_undefined"]
    record = read_json(tmp_path / "search.json")
    assert record["models"]["ebt"]["chosen"] == asdict(Example("ebt", 1e-3))
    assert len(run.calls) == 3 + 1


def test_files_record_the_fingerprint_and_a_matching_rerun_does_no_work(tmp_path):
    execute(tmp_path, Recorder())
    assert (
        read_json(tmp_path / "search.json")["models"]["ebt"]["code_fingerprint"] == "current code"
    )
    assert read_json(tmp_path / "ebt-seed0.json")["code"] == {
        "fingerprint": "current code",
        "reused_stale_files": [],
        "checkpoint": (tmp_path / "checkpoints" / "ebt-seed0.pt").resolve().as_posix(),
    }
    again = Recorder()
    execute(tmp_path, again)
    assert again.calls == []


@pytest.mark.parametrize("stored", ["other code", None])
def test_a_stale_file_stops_the_run_before_any_work(tmp_path, stored):
    """A different fingerprint and a missing one (files written before fingerprints) are
    both stale."""
    execute(tmp_path, Recorder(), Provenance("other code"))
    path = tmp_path / "search" / "ebt" / "learning_rate=0.001_model=ebt.json"
    entry = read_json(path)
    if stored is None:
        del entry["code_fingerprint"]
    write_json(path, entry)

    run = Recorder()
    with pytest.raises(StaleResultsError, match=re.escape("learning_rate=0.001_model=ebt.json")):
        execute(tmp_path, run)
    assert run.calls == []


def test_explicit_reuse_of_stale_files_is_recorded_in_the_new_files(tmp_path):
    execute(tmp_path, Recorder(), Provenance("other code"))
    (tmp_path / "ebt-seed0.json").unlink()

    run = Recorder()
    execute(tmp_path, run, Provenance("current code", reuse_stale=True))

    assert run.calls == [(Example("ebt", 3e-4), 0, True)]
    stale = [f"search/ebt/{candidate_name(c)}.json" for c in GRID]
    assert read_json(tmp_path / "search.json")["reused_stale_files"] == sorted(stale)
    assert read_json(tmp_path / "ebt-seed0.json")["code"]["reused_stale_files"] == sorted(stale)


def test_final_stage_alone_treats_a_stale_choice_as_stale(tmp_path):
    execute(tmp_path, Recorder(), Provenance("other code"))
    (tmp_path / "ebt-seed0.json").unlink()
    with pytest.raises(StaleResultsError, match=re.escape("search.json#models/ebt/chosen")):
        execute(tmp_path, Recorder(), stage="final")

    execute(tmp_path, Recorder(), Provenance("current code", reuse_stale=True), stage="final")
    stale = [f"search/ebt/{candidate_name(c)}.json" for c in GRID]
    assert read_json(tmp_path / "ebt-seed0.json")["code"]["reused_stale_files"] == sorted(
        ["search.json#models/ebt/chosen", *stale]
    )


TWO = {
    "ebt": GRID,
    "denoiser": [Example("denoiser", 1e-3), Example("denoiser", 1e-4)],
}


def execute_two(path, run, models, provenance=CURRENT, stage="all"):
    runner.execute(
        path, stage, models, TWO.__getitem__, Example, run, 100, (0,), provenance, path / "ckpt"
    )


def test_partial_search_leaves_old_search_of_other_models_stale(tmp_path):
    """search.json from earlier code: `tried` lists without files and without fingerprints.
    Searching one model records the current fingerprint only for that model; a final stage
    for both models then stops before any work unless stale reuse is allowed, and with it
    records the reused entries and choice of the other model."""
    old = {
        "models": {
            model: {
                "tried": [
                    {"candidate": asdict(c), "selection_value": c.learning_rate} for c in grid
                ],
                "chosen": asdict(min(grid, key=lambda c: c.learning_rate)),
            }
            for model, grid in TWO.items()
        }
    }
    write_json(tmp_path / "search.json", old)

    execute_two(tmp_path, Recorder(), ["ebt"], stage="search")
    record = read_json(tmp_path / "search.json")
    assert record["models"]["ebt"]["code_fingerprint"] == "current code"
    assert "code_fingerprint" not in record["models"]["denoiser"]
    assert "code_fingerprint" not in record

    run = Recorder()
    with pytest.raises(StaleResultsError) as error:
        execute_two(tmp_path, run, ["ebt", "denoiser"], stage="final")
    assert run.calls == []
    old_denoiser = [
        f"search.json#models/denoiser/tried/{candidate_name(c)}" for c in TWO["denoiser"]
    ]
    for label in [*old_denoiser, "search.json#models/denoiser/chosen"]:
        assert label in str(error.value)
    assert "models/ebt/" not in str(error.value)

    run = Recorder()
    execute_two(
        tmp_path, run, ["ebt", "denoiser"], Provenance("current code", reuse_stale=True), "final"
    )
    assert run.calls == [(Example("ebt", 3e-4), 0, True), (Example("denoiser", 1e-4), 0, True)]
    assert read_json(tmp_path / "ebt-seed0.json")["code"]["reused_stale_files"] == []
    assert read_json(tmp_path / "denoiser-seed0.json")["code"]["reused_stale_files"] == sorted(
        [*old_denoiser, "search.json#models/denoiser/chosen"]
    )


def test_a_final_file_from_other_code_is_recomputed_not_reused(tmp_path):
    execute(tmp_path, Recorder(), Provenance("other code"))
    run = Recorder()
    execute(tmp_path, run, Provenance("current code", reuse_stale=True), stage="final")
    assert run.calls == [(Example("ebt", 3e-4), 0, True)]
    assert read_json(tmp_path / "ebt-seed0.json")["code"]["fingerprint"] == "current code"
