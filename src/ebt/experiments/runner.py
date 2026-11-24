"""Search and final stages shared by the experiments: written as they finish, resumable.

A search tries every configuration of a model once with the search seed, on validation data,
and chooses one by `choose`. The final stage trains the chosen configuration with every
reported seed and evaluates it on test data. Every configuration and every final run is
written as soon as it finishes, and a rerun skips what is already on disk.

Every final run also saves its trained parameters to a checkpoint (ebt.checkpoints).

Every file records the fingerprint of the code that wrote it (ebt.fingerprint): each search
entry, each model's choice in search.json, and each result file. A search entry or a choice with
another fingerprint, or with none, is stale; this includes entries that exist only in a
search.json `tried` list. A search records the current fingerprint only for the models it
searches; the entries and choices of other models keep theirs. Before any work, `execute` stops
and lists the stale entries and choices the stage would use, unless reuse is allowed explicitly.
Reused stale entries are then listed in search.json (search files) and in every result file that
depends on them. A final run is never reused from other code: it is skipped only if its result
file and its checkpoint both exist and both carry the current fingerprint, and recomputed
otherwise.

The final stage starts only from a complete search: every configuration of the search space is
on disk, and the recorded choice equals a fresh `choose` over all of them.
"""

import dataclasses
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from ebt.checkpoints import checkpoint_fingerprint, checkpoint_path, recorded_path, save_checkpoint
from ebt.results import RunResult, read_json, write_json, write_result

Run = Callable[[Any, int, bool], tuple[RunResult, tuple[str, float], nn.Module]]
"""run(candidate, seed, with_test) -> (result, (selection metric name, value), trained module)."""

SELECTION_RULE = (
    "minimum of each model's own validation objective, compared exactly as floats; among "
    "exactly equal values the first configuration in grid order; an undefined value (null, "
    "from NaN, an infinity or a run stopped by a non-finite value) ranks after every defined "
    "value and is chosen only if every value is undefined"
)


@dataclass(frozen=True)
class Provenance:
    """The fingerprint of the current code and whether stale files may be reused."""

    fingerprint: str
    reuse_stale: bool = False


REUSE_HELP = (
    "reuse result files written by other code instead of stopping; the reuse is recorded in "
    "search.json and in every result file written afterwards"
)


class StaleResultsError(RuntimeError):
    """Result files on disk were written by other code (see the module docstring)."""


class IncompleteSearchError(RuntimeError):
    """The final stage found a search with missing configurations or a stale choice."""


def status(message: str) -> None:
    """One status line with the wall-clock time, flushed at once."""
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def candidate_name(candidate: Any) -> str:
    """File name of one search configuration: its fields that are not None, sorted."""
    values = dataclasses.asdict(candidate)
    return "_".join(f"{key}={values[key]}" for key in sorted(values) if values[key] is not None)


def same_configuration(first: dict, second: dict) -> bool:
    """Whether two configurations agree on every field that is set (not None)."""

    def fields(values: dict) -> dict:
        return {key: value for key, value in values.items() if value is not None}

    return fields(first) == fields(second)


def choose(entries: Sequence[dict]) -> dict:
    """The entry with the smallest `selection_value` (method.md section 16.6).

    Values are compared exactly as floats. Among exactly equal values the first entry, in grid
    order, is kept. An undefined value (None, NaN or an infinity) ranks after every defined
    value.
    """

    def rank(value: float | None) -> float:
        return math.inf if value is None or not math.isfinite(value) else value

    best = entries[0]
    for entry in entries[1:]:
        if rank(entry["selection_value"]) < rank(best["selection_value"]):
            best = entry
    return best


def stored_fingerprint(payload: dict) -> str | None:
    """The fingerprint recorded in a search entry, search.json or a result file, if any."""
    if "code_fingerprint" in payload:
        return payload["code_fingerprint"]
    return payload.get("code", {}).get("fingerprint")


def search_path(output: Path, model: str, candidate: Any) -> Path:
    return output / "search" / model / (candidate_name(candidate) + ".json")


def final_path(output: Path, model: str, seed: int) -> Path:
    return output / f"{model}-seed{seed}.json"


def stale(paths: Sequence[Path], fingerprint: str) -> list[Path]:
    """The existing files among `paths` whose fingerprint differs from `fingerprint`."""
    return [
        path
        for path in paths
        if path.exists() and stored_fingerprint(read_json(path)) != fingerprint
    ]


def evaluate(
    run: Run, candidate: Any, seed: int
) -> tuple[RunResult | None, str | None, float | None, str | None]:
    """Run one search configuration; a non-finite outcome gives value None and its reason.

    A run stopped by FloatingPointError (a non-finite thinking gradient) has no result.
    """
    try:
        result, (name, value), _ = run(candidate, seed, False)
    except FloatingPointError as error:
        return None, None, None, f"FloatingPointError: {error}"
    if math.isfinite(value):
        return result, name, value, None
    return result, name, None, f"selection value {value}"


def search(
    output: Path,
    models: Sequence[str],
    space: Callable[[str], list],
    run: Run,
    seed: int,
    provenance: Provenance,
) -> dict:
    """Try every configuration of `models` on validation data with `seed`.

    Each configuration is written to search/<model>/ as soon as it finishes and skipped on a
    rerun. A configuration whose selection value is not finite is recorded with value null and
    the reason in `selection_value_undefined`; it ranks last. search.json then collects every
    model's configurations and choice, with the current fingerprint for the models searched in
    this call; the entries of other models are kept unchanged, with their own fingerprints.
    """
    summary = output / "search.json"
    record = read_json(summary) if summary.exists() else {"models": {}}
    record["selection_rule"] = SELECTION_RULE
    record.pop("code_fingerprint", None)  # written by earlier code for the whole file
    reused = set(record.get("reused_stale_files", []))
    for model in models:
        reused = {name for name in reused if not name.startswith(f"search/{model}/")}
        tried = []
        for candidate in space(model):
            path = search_path(output, model, candidate)
            if path.exists():
                entry = read_json(path)
                if stored_fingerprint(entry) != provenance.fingerprint:
                    reused.add(path.relative_to(output).as_posix())
                tried.append(entry)
                continue
            start = time.perf_counter()
            result, name, value, undefined = evaluate(run, candidate, seed)
            entry = {
                "candidate": dataclasses.asdict(candidate),
                "code_fingerprint": provenance.fingerprint,
                "selection_metric": name,
                "selection_value": value,
                "selection_value_undefined": undefined,
                "validation_metrics_for_information": (
                    None if result is None else result.metrics["validation"]
                ),
            }
            write_json(path, entry)
            tried.append(entry)
            elapsed = time.perf_counter() - start
            shown = undefined if value is None else f"{value:.6g}"
            status(f"search {model}/{path.name} {name}={shown} ({elapsed:.1f} s)")
        record["models"][model] = {
            "tried": tried,
            "chosen": choose(tried)["candidate"],
            "code_fingerprint": provenance.fingerprint,
        }
        record["reused_stale_files"] = sorted(reused)
        write_json(summary, record)
    return record


def search_entries(
    output: Path, record: dict, model: str, space: Callable
) -> tuple[list[tuple[str, dict]], list[Path]]:
    """The search entries of `model` for every configuration of `space`, with a label each, and
    the files of the configurations not found.

    A configuration is found if its file under search/<model>/ exists (label: that path relative
    to `output`) or, for searches written before those files existed, if search.json lists it
    among the model's `tried` entries (label: search.json#models/<model>/tried/<name>).
    Configurations are matched with `same_configuration`.
    """
    listed = record.get("models", {}).get(model, {})
    found, missing = [], []
    for candidate in space(model):
        path = search_path(output, model, candidate)
        if path.exists():
            found.append((path.relative_to(output).as_posix(), read_json(path)))
            continue
        wanted = dataclasses.asdict(candidate)
        matches = [e for e in listed.get("tried", []) if same_configuration(e["candidate"], wanted)]
        if matches:
            label = f"search.json#models/{model}/tried/{candidate_name(candidate)}"
            found.append((label, matches[0]))
        else:
            missing.append(path)
    return found, missing


def check_search(
    output: Path, record: dict, models: Sequence[str], space: Callable, fingerprint: str
) -> list[str]:
    """Raise IncompleteSearchError unless, for every model, every configuration of `space` is
    found (`search_entries`) and the choice recorded in `record` equals `choose` over all of
    them. Returns the labels of the stale entries and choices: an entry or a model's choice
    whose fingerprint differs from `fingerprint` or is missing (the choice of model m is
    labelled search.json#models/<m>/chosen).
    """
    labels = []
    for model in models:
        listed = record.get("models", {}).get(model, {})
        found, missing = search_entries(output, record, model, space)
        if missing:
            listing = "\n".join(f"  {path}" for path in missing)
            raise IncompleteSearchError(
                f"the search of {model} is incomplete; {len(missing)} configurations are not on "
                f"disk:\n{listing}\nRun the search stage first."
            )
        fresh = choose([entry for _, entry in found])["candidate"]
        recorded = listed.get("chosen")
        if recorded is None or not same_configuration(recorded, fresh):
            raise IncompleteSearchError(
                f"search.json records {recorded} as the choice for {model}, but the "
                f"configurations on disk give {fresh}. Run the search stage again to rewrite "
                "search.json."
            )
        labels += [label for label, entry in found if stored_fingerprint(entry) != fingerprint]
        if listed.get("code_fingerprint") != fingerprint:
            labels.append(f"search.json#models/{model}/chosen")
    return labels


def depends_on(label: str, model: str) -> bool:
    """Whether the stale entry or choice `label` belongs to `model`'s search."""
    return label.startswith((f"search/{model}/", f"search.json#models/{model}/"))


def final(
    output: Path,
    record: dict,
    models: Sequence[str],
    space: Callable[[str], list],
    candidate_type: type,
    run: Run,
    seeds: Sequence[int],
    provenance: Provenance,
    checkpoints: Path,
) -> None:
    """Train each model with its chosen configuration for each of `seeds`, with test data.

    Checks the search first (`check_search`); if any entry or choice it uses is stale, raises
    StaleResultsError before any work unless `provenance.reuse_stale` is set. Each run is
    written as soon as it finishes, with the code fingerprint and the labels of the stale
    entries and choice of its own model's search. Its trained module is saved to
    `checkpoints`/<model>-seed<seed>.pt before the result file. A run is skipped only if its
    result file holds the chosen configuration (fields set to None are ignored) and the current
    fingerprint, and its checkpoint exists with the current fingerprint; otherwise it is rerun
    and both files are replaced.
    """
    found = check_search(output, record, models, space, provenance.fingerprint)
    if found and not provenance.reuse_stale:
        raise_stale(found)
    for model in models:
        chosen = record["models"][model]["chosen"]
        candidate = candidate_type(**chosen)
        for seed in seeds:
            path = final_path(output, model, seed)
            checkpoint = checkpoint_path(checkpoints, model, seed)
            if done(path, checkpoint, chosen, provenance.fingerprint):
                continue
            start = time.perf_counter()
            result, _, trained = run(candidate, seed, True)
            save_checkpoint(checkpoint, trained, provenance.fingerprint, result)
            result.code = {
                "fingerprint": provenance.fingerprint,
                "reused_stale_files": sorted(n for n in found if depends_on(n, model)),
                "checkpoint": recorded_path(checkpoint),
            }
            write_result(path, result)
            status(f"final {path.name} ({time.perf_counter() - start:.1f} s)")


def raise_stale(found: Sequence[object]) -> None:
    listing = "\n".join(f"  {item}" for item in found)
    raise StaleResultsError(
        f"{len(found)} search entries or choices were written by other code (fingerprint "
        f"differs or is missing):\n{listing}\nMove the files away to recompute them, or pass "
        "--reuse-stale to reuse them; reuse is recorded in the new files."
    )


def done(path: Path, checkpoint: Path, chosen: dict, fingerprint: str) -> bool:
    """Whether a final run is finished with the current code (see `final`)."""
    if not path.exists():
        return False
    stored = read_json(path)
    return (
        same_configuration(stored["config"]["candidate"], chosen)
        and stored_fingerprint(stored) == fingerprint
        and checkpoint_fingerprint(checkpoint) == fingerprint
    )


def execute(
    output: Path,
    stage: str,
    models: Sequence[str],
    space: Callable[[str], list],
    candidate_type: type,
    run: Run,
    search_seed: int,
    seeds: Sequence[int],
    provenance: Provenance,
    checkpoints: Path,
) -> None:
    """Run `stage` ("search", "final" or "all") after checking every file it would reuse.

    Final runs save their trained parameters to `checkpoints`.

    Raises StaleResultsError before any work if the stage would use a stale search entry or
    choice and `provenance.reuse_stale` is False: for "search" and "all", the existing search
    files of `models`; for "final", every entry and choice `final` uses (`check_search`). Final
    result files are not reused from other code (`final`), so they are not part of this check.
    """
    summary = output / "search.json"
    if stage in ("search", "all"):
        paths = [search_path(output, model, c) for model in models for c in space(model)]
        found = stale(paths, provenance.fingerprint)
        if found and not provenance.reuse_stale:
            raise_stale(found)
        record = search(output, models, space, run, search_seed, provenance)
    else:
        if not summary.exists():
            raise IncompleteSearchError(f"{summary} does not exist. Run the search stage first.")
        record = read_json(summary)
    if stage in ("final", "all"):
        final(output, record, models, space, candidate_type, run, seeds, provenance, checkpoints)
