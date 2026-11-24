"""Budget rule of the text experiment (method.md section 18.7) and the measurements it needs.

usage: python -m tools.text_budget OUTPUT_DIR [--measure-command COMMAND]
       [--high-watermark-ratio 1.0] [--low-watermark-ratio 0.8] [--timeout 1800]

Applies items 1 to 4 of the budget rule of method.md section 18.7 (`decide`) and measures, one
setting per process, every setting the rule asks for until it can decide (`run_measurement`).

Outputs in OUTPUT_DIR: measurements.jsonl and oom.jsonl (appended to, so a second run resumes),
memory.json (the free memory, read once, and the time limit), timing.log and measure-stderr.log,
section-18.8.md (the measurement table and the application of the rule) and, if a setting fits,
choice.json. Exit 0 if a setting is chosen, 2 if nothing fits, 1 on any failure or usage error.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SIZES = [(384, 6), (256, 6), (128, 4)]
"""(width, depth), tried from the largest (item 2)."""
SETTINGS = [(256, 64), (256, 32), (128, 64), (128, 32)]
"""(sequence length, batch), tried in this order for each size (item 3)."""
LENGTHS = [1000, 2000, 5000]
STAGE_BUDGET = 8 * 3600
"""Wall-clock budget of v1.0 for each stage (the search; the final runs), in seconds, on one
Apple M5 Mac with 16 GB (method.md section 18.7). Later versions record their own budget."""
CONFIGS, FINALS, SAMPLE_STEPS = 4, 5, 500
"""Search configurations per model, final runs per model, characters per generated sample."""
MEMORY_FRACTION = 0.75

TIMEOUT = 1800
"""Time limit of one measurement process, in seconds (method.md section 18.7)."""
OUT_OF_MEMORY, OVER_TIME = "out of memory", "timeout"
METAL_OUT_OF_MEMORY = ("Insufficient Memory", "kIOGPUCommandBufferCallbackErrorOutOfMemory")
"""Texts of a Metal driver abort on a memory shortage."""

Record = dict[str, Any]
"""One measurement: the fields of `key` and the measured times, memory and parameter count."""


def spaced(number: int) -> str:
    """1 066 689 rather than 1,066,689, as in method.md."""
    return f"{number:,}".replace(",", " ")


def key(dim: int, depth: int, seq: int, batch: int, model: str, layout: str) -> str:
    return f"{dim}x{depth} S{seq} B{batch} {model} {layout}"


def record_key(record: Record) -> str:
    return key(
        record["dim"],
        record["depth"],
        record["seq"],
        record["batch"],
        record["model"],
        record["layout"],
    )


def projection_seconds(baseline: Record, ebt: Record, length: int) -> tuple[float, float]:
    """Projected seconds of the search and of the final runs at training length `length`.

    The samples of seed 0 are one generation per model: SAMPLE_STEPS characters, both samples in
    one batch of 2 rows (`text_lm.generate`), the rows of the measured generation time. The
    baseline's time per character is its measured value if its record has one; otherwise the
    EBT's, an upper bound (one forward pass per character against the EBT's 4 forward and 4
    backward passes on the same rows)."""
    run_b = baseline["step_s"] * length + 10 * baseline["validation_eval_s"]
    run_e = ebt["step_mean_s"] * length + 10 * ebt["validation_eval_s"]
    search = CONFIGS * (run_b + run_e)
    ebt_character = ebt["generation_s_per_character"]
    baseline_character = baseline.get("generation_s_per_character", ebt_character)
    samples = SAMPLE_STEPS * ebt_character + SAMPLE_STEPS * baseline_character
    finals = (
        FINALS * (run_b + run_e + 2 * baseline["validation_eval_s"] + 2 * ebt["validation_eval_s"])
        + samples
    )
    return search, finals


def projection(baseline: Record, ebt: Record, length: int) -> tuple[float, float]:
    """Projected hours of the search and of the final runs at training length `length`."""
    search, finals = projection_seconds(baseline, ebt, length)
    return search / 3600, finals / 3600


def failure_reason(failed: Mapping[str, str], name: str) -> str:
    """OUT_OF_MEMORY or OVER_TIME, the reason recorded for the key `name` in `failed`."""
    return failed[name]


def failure_text(reason: str) -> str:
    return "ran out of memory" if reason == OUT_OF_MEMORY else "exceeded the time limit"


@dataclass
class Decision:
    """Outcome of `decide`: `missing` is the next measurement needed, as the keyword arguments
    of `key`; otherwise `choice` is the chosen setting, or None if nothing fits."""

    limit_gib: float
    lines: list[str] = field(default_factory=list)
    choice: dict[str, Any] | None = None
    missing: dict[str, Any] | None = None


_MISSING = object()


def decide(records: Mapping[str, Record], failed: Mapping[str, str], free_gib: float) -> Decision:
    """Apply items 1 to 4 of method.md section 18.7 to `records` (keyed by `key`) and to `failed`,
    the keys of the measurements that did not fit with their reason (`failure_reason`).

    Sizes are tried from the largest, and for each size the settings in the order of item 3. For
    each setting the EBT is measured in the split layout, then in the full layout, then the
    baseline; if the split layout does not fit, the setting fails without further measurements.
    A measurement over the time limit counts exactly as one out of memory. Memory: at most
    MEMORY_FRACTION of `free_gib`, the limit included. Time: at most STAGE_BUDGET seconds for the
    search and for the final runs, the limit included. The first size and setting that meet items
    2 and 4 are chosen. The result is the choice, "nothing fits" (choice None), or the next
    measurement the rule needs (`missing`)."""
    limit = MEMORY_FRACTION * free_gib
    decision = Decision(limit_gib=limit)
    lines = decision.lines

    def lookup(**fields: Any) -> Record | object | None:
        name = key(**fields)
        if name in records:
            return records[name]
        if name in failed:
            return None
        decision.missing = fields
        return _MISSING

    for dim, depth in SIZES:
        for seq, batch in SETTINGS:
            label = f"{dim} × {depth}, S {seq}, batch {batch}"
            setting = {"dim": dim, "depth": depth, "seq": seq, "batch": batch}
            split = lookup(**setting, model="ebt", layout="split")
            if split is _MISSING:
                return decision
            if split is None:
                why = failure_text(
                    failure_reason(failed, key(**setting, model="ebt", layout="split"))
                )
                lines.append(f"- {label}: the EBT {why} (split layout).")
                continue
            full = lookup(**setting, model="ebt", layout="full")
            if full is _MISSING:
                return decision
            layouts = [r for r in (split, full) if r is not None]
            ebt = min(layouts, key=lambda r: r["step_k4_s"])
            times = ", ".join(f"{r['layout']} {r['step_k4_s']:.3f} s" for r in layouts)
            if full is None:
                reason = failure_reason(failed, key(**setting, model="ebt", layout="full"))
                times += ", full out of memory" if reason == OUT_OF_MEMORY else ", full over time"
            if ebt["peak_driver_k4_gib"] > limit:
                lines.append(
                    f"- {label}: faster layout {ebt['layout']} (k = 4: {times}); "
                    f"{ebt['peak_driver_k4_gib']:.2f} GiB at k = 4, above {limit:.2f} GiB."
                )
                continue
            baseline = lookup(**setting, model="transformer", layout="split")
            if baseline is _MISSING:
                return decision
            if baseline is None:
                name = key(**setting, model="transformer", layout="split")
                lines.append(
                    f"- {label}: the baseline {failure_text(failure_reason(failed, name))}."
                )
                continue
            fits = []
            for length in LENGTHS:
                search_s, finals_s = projection_seconds(baseline, ebt, length)
                fits.append(
                    (
                        length,
                        search_s / 3600,
                        finals_s / 3600,
                        search_s <= STAGE_BUDGET and finals_s <= STAGE_BUDGET,
                    )
                )
            shown = "; ".join(
                f"T = {spaced(t)}: search {s:.2f} h, finals {f:.2f} h" for t, s, f, _ in fits
            )
            lines.append(
                f"- {label}: faster layout {ebt['layout']} (k = 4: {times}); "
                f"{ebt['peak_driver_k4_gib']:.2f} GiB at k = 4, within {limit:.2f} GiB; "
                f"mean EBT step {ebt['step_mean_s']:.3f} s, baseline "
                f"{baseline['step_s']:.3f} s; {shown}."
            )
            good = [t for t, _, _, ok in fits if ok]
            if good:
                length = max(good)
                search, finals = next((s, f) for t, s, f, _ in fits if t == length)
                decision.choice = {
                    "dim": dim,
                    "depth": depth,
                    "sequence_length": seq,
                    "batch_size": batch,
                    "layout": ebt["layout"],
                    "train_steps": length,
                    "validate_every": length // 10,
                    "search_hours": search,
                    "final_hours": finals,
                    "ebt_parameters": ebt["parameters"],
                    "baseline_parameters": baseline["parameters"],
                }
                return decision
            lines[-1] += " T = 1 000 does not fit."
    return decision


def seconds(value: float) -> str:
    return spaced(int(value)) if value == int(value) else f"{value:g}"


def section(
    memory: Mapping[str, Any],
    decision: Decision,
    records: Mapping[str, Record],
    failed: Mapping[str, str],
) -> str:
    """Text of the measurement for method.md section 18.8. The time limit of a measurement is
    memory["timeout_s"] (TIMEOUT if absent); the date is memory["date"]."""
    limit, choice = decision.limit_gib, decision.choice
    timeout = seconds(memory.get("timeout_s", TIMEOUT))
    measured_on = memory["date"]
    rows = [
        "| size | S | batch | model | layout | parameters | step k=2 (s) | step k=3 (s) | "
        "step k=4 (s) | step (s) | peak k=4 (GiB) | validation eval (s) | generation (s/char) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records.values():
        if r["model"] == "transformer":
            rows.append(
                f"| {r['dim']} × {r['depth']} | {r['seq']} | {r['batch']} | transformer | - "
                f"| {r['parameters']:,} | | | | {r['step_s']:.4f} | "
                f"{r['peak_driver_gib']:.2f} | {r['validation_eval_s']:.2f} | |"
            )
        else:
            rows.append(
                f"| {r['dim']} × {r['depth']} | {r['seq']} | {r['batch']} | EBT | "
                f"{r['layout']} | {r['parameters']:,} | {r['step_k2_s']:.4f} | "
                f"{r['step_k3_s']:.4f} | {r['step_k4_s']:.4f} | {r['step_mean_s']:.4f} | "
                f"{r['peak_driver_k4_gib']:.2f} | {r['validation_eval_s']:.2f} | "
                f"{r['generation_s_per_character']:.4f} |"
            )
    lines = [
        "",
        f"Measurement of the current model (sections 4.1 and 18.3, corrections 13.1 to 13.4), "
        f"{measured_on}, no other training process running. Before the measurements: "
        f"`memory_pressure` reported {memory['free_percent']}% free, that is "
        f"{memory['free_gib']:.2f} GiB of {memory['total_gib']:.0f} GiB, so the memory limit is "
        f"{limit:.2f} GiB; load averages {memory['loadavg']}. Same procedure as above; the "
        "measurements follow the order of item 3 and stop at the first size and setting that "
        "meet items 2 and 4. A process that ran out of MPS memory, was killed by the system or "
        "was ended by a Metal out-of-memory abort is recorded as out of memory; one that ran "
        f"longer than {timeout} s is stopped and recorded as over the time limit. "
        "Measured and applied with `python -m tools.text_budget` "
        "(`tools/text_budget.py`, one process per setting with "
        "`tools/text_step_measurement.py`).",
        "",
        *rows,
        "",
    ]
    memory_failures = sorted(n for n in failed if failure_reason(failed, n) == OUT_OF_MEMORY)
    time_failures = sorted(n for n in failed if failure_reason(failed, n) == OVER_TIME)
    if memory_failures:
        lines += ["Out of memory: " + "; ".join(memory_failures) + ".", ""]
    if time_failures:
        lines += [
            f"Over the time limit of {timeout} s: " + "; ".join(time_failures) + ".",
            "",
        ]
    lines += ["Application of section 18.7 (item 3):", "", *decision.lines, ""]
    if choice:
        departures = []
        if (choice["sequence_length"], choice["batch_size"]) != (256, 64):
            departures.append(
                f"S {choice['sequence_length']} and batch {choice['batch_size']} "
                "instead of S 256 and batch 64, forced by the rows above"
            )
        if (choice["dim"], choice["depth"]) != (384, 6):
            departures.append(
                "a size smaller than the authors' xxs (384 × 6), forced by the rows above"
            )
        lines += [
            f"Chosen: width {choice['dim']}, depth {choice['depth']} "
            f"({spaced(choice['ebt_parameters'])} EBT and "
            f"{spaced(choice['baseline_parameters'])} baseline parameters), "
            f"S = {choice['sequence_length']}, batch {choice['batch_size']}, "
            f"T = {spaced(choice['train_steps'])}, {choice['layout']} layout, validation every "
            f"{choice['validate_every']} steps; projected search {choice['search_hours']:.2f} h "
            f"and final runs {choice['final_hours']:.2f} h, each within the v1.0 budget of "
            f"{STAGE_BUDGET // 3600} h per stage.",
            "",
            "Departures from Table D.2: " + ("; ".join(departures) if departures else "none") + ".",
            "",
        ]
    else:
        lines += ["No size and setting meet items 2 and 4; the search does not start.", ""]
    return "\n".join(lines)


def read_lines(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


STARTED = time.perf_counter()
"""The start of the tool; log lines record the seconds elapsed since then."""


def elapsed() -> str:
    return f"{time.perf_counter() - STARTED:.1f} s"


def log(out: Path, message: str) -> None:
    line = f"{elapsed()} {message}"
    print(line, flush=True)
    with open(out / "timing.log", "a") as handle:
        handle.write(line + "\n")


def free_memory() -> dict[str, Any]:
    """Free memory from `memory_pressure` and `sysctl` (macOS), with the date and the load."""
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    percent = int(re.search(r"free percentage: (\d+)%", pressure).group(1))
    total = int(
        subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    )
    load = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout
    total_gib = total / 2**30
    return {
        "date": time.strftime("%Y-%m-%d"),
        "free_percent": percent,
        "total_gib": total_gib,
        "free_gib": percent / 100 * total_gib,
        "loadavg": load.strip(),
    }


def is_out_of_memory(returncode: int, stderr: str) -> bool:
    """Out of memory: an out-of-memory error message, a kill by the system (exit 137, signal 9),
    or SIGABRT (exit 134, signal 6) with a text of METAL_OUT_OF_MEMORY in the error output."""
    if returncode in (137, -9) or "out of memory" in stderr.lower():
        return True
    return returncode in (134, -6) and any(text in stderr for text in METAL_OUT_OF_MEMORY)


def read_failed(path: Path) -> dict[str, str]:
    """Key and reason of every line of oom.jsonl; lines without a reason are out of memory."""
    return {r["key"]: r.get("reason", OUT_OF_MEMORY) for r in read_lines(path)}


class DoesNotFit(Exception):
    """A measurement process ran out of memory or over the time limit; `reason` is OUT_OF_MEMORY
    or OVER_TIME."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.reason = reason


def run_measurement(
    out: Path,
    command: list[str],
    env: Mapping[str, str],
    fields: Mapping[str, Any],
    timeout: float = TIMEOUT,
) -> Record:
    """Measure one setting in its own process (the command followed by --dim, --depth, --seq,
    --batch, --model and --layout) and return its record, appended to measurements.jsonl.

    Raises DoesNotFit if the process ran out of memory (`is_out_of_memory`) or over `timeout`
    seconds, after appending the key and the reason to oom.jsonl; exits with 1 on any other
    failure. The process runs in a new session, and on timeout its whole process group is killed
    (method.md section 18.7 explains why such a setting cannot fit)."""
    name = key(**fields)
    log(out, f"start {name}")
    arguments = [f"--{field}={value}" for field, value in fields.items()]
    process = subprocess.Popen(
        command + arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        reason = None if process.returncode == 0 else OUT_OF_MEMORY
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        reason = OVER_TIME
    with open(out / "measure-stderr.log", "a") as handle:
        handle.write(f"--- {elapsed()} {name} exit {process.returncode}\n")
        handle.write(stderr)
    if reason == OUT_OF_MEMORY and not is_out_of_memory(process.returncode, stderr):
        log(out, f"FAILED {name} exit {process.returncode}")
        sys.exit(1)
    if reason is not None:
        with open(out / "oom.jsonl", "a") as handle:
            line = {"key": name, "exit": process.returncode, "reason": reason}
            handle.write(json.dumps(line) + "\n")
        shown = "OOM" if reason == OUT_OF_MEMORY else f"TIMEOUT after {timeout} s"
        log(out, f"{shown} {name} (exit {process.returncode})")
        raise DoesNotFit(name, reason)
    record = json.loads(stdout.strip().splitlines()[-1])
    with open(out / "measurements.jsonl", "a") as handle:
        handle.write(json.dumps(record) + "\n")
    log(out, f"done {name}")
    return record


class Parser(argparse.ArgumentParser):
    """Usage errors exit with 1: exit 2 means that nothing fits."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = Parser(description=__doc__.split("\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--measure-command",
        default=f"{shlex.quote(sys.executable)} -u -m tools.text_step_measurement --device mps",
        help="command that measures one setting (tools.text_step_measurement)",
    )
    parser.add_argument("--high-watermark-ratio", default="1.0")
    parser.add_argument("--low-watermark-ratio", default="0.8")
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="seconds per measurement")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read the free memory once, then measure what `decide` asks for until it decides."""
    args = parse(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    state = out / "memory.json"
    memory = json.loads(state.read_text()) if state.exists() else free_memory()
    memory["timeout_s"] = args.timeout
    state.write_text(json.dumps(memory))
    command = shlex.split(args.measure_command)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_MPS_HIGH_WATERMARK_RATIO=args.high_watermark_ratio,
        PYTORCH_MPS_LOW_WATERMARK_RATIO=args.low_watermark_ratio,
    )
    records = {record_key(r): r for r in read_lines(out / "measurements.jsonl")}
    failed: MutableMapping[str, str] = read_failed(out / "oom.jsonl")
    decision = decide(records, failed, memory["free_gib"])
    log(
        out,
        f"free memory {memory['free_gib']:.2f} GiB ({memory['free_percent']}%), "
        f"limit {decision.limit_gib:.2f} GiB, load {memory['loadavg']}",
    )
    measured = set()
    while decision.missing is not None:
        name = key(**decision.missing)
        if name in measured:
            log(out, f"FAILED the rule asked again for {name}, which was measured")
            return 1
        measured.add(name)
        try:
            record = run_measurement(out, command, env, decision.missing, args.timeout)
        except DoesNotFit as failure:
            failed[name] = failure.reason
        else:
            records[record_key(record)] = record
        decision = decide(records, failed, memory["free_gib"])
    (out / "section-18.8.md").write_text(section(memory, decision, records, failed))
    if decision.choice is None:
        log(out, "nothing fits")
        return 2
    (out / "choice.json").write_text(json.dumps(decision.choice, indent=1))
    log(out, f"CHOSEN {decision.choice}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
