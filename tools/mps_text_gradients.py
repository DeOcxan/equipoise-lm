"""MPS against CPU for the S2 parameter gradients of the small text EBT (docs/numerics.md).

usage: python -m tools.mps_text_gradients measure [--device mps] > DIAGNOSIS_JSONL
       python -m tools.mps_text_gradients decide DIAGNOSIS_JSONL [--factor 2.5]

measure computes the gradients in float64 on CPU as the reference, in float32 on CPU and in
float32 on the device (`gradients`), and compares them (`errors`). decide derives whether the
difference is float32 rounding and the tolerance of the test (`decide`).

Outputs: measure prints one JSON line per layout (full, split); decide prints the comparison and a
line starting with DECISION, and exits with 0 if the difference is float32 rounding, 1 otherwise.
"""

import argparse
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

FACTOR = 2.5
FLOAT32_DEFAULT_ATOL = 1e-5
LAYOUTS = ("full", "split")


def gradients(
    device: str, layout: str, dtype: "torch.dtype"
) -> tuple[list[str], list["torch.Tensor"]]:
    """The names and S2 parameter gradients of the small text EBT, as float64 tensors on the CPU:
    seeds 19 and 20, V 9, D 64, depth 2, S2 with 3 steps and energy-scaled α 5, the computation
    of `tests/test_mps.py: test_mps_text_energy_s2_gradients_match_cpu`."""
    import torch

    from ebt.config import ThinkingConfig, TrainingMode
    from ebt.core.thinking import ThinkingLoop
    from ebt.models.text_transformer import TextEnergy, TextTransformerConfig
    from ebt.tasks.text import cross_entropy

    torch.manual_seed(19)
    model = TextEnergy(TextTransformerConfig(vocabulary=9, dim=64, depth=2, layout=layout))
    model = model.to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(20)
    context = torch.randint(0, 9, (3, 6), generator=generator).to(device)
    target = torch.randint(0, 9, (3, 6), generator=generator).to(device)
    initial = torch.randn(3, 6, 9, generator=generator).to(device=device, dtype=dtype)
    loop = ThinkingLoop(
        ThinkingConfig(
            mode=TrainingMode.S2, step_size=5.0, energy_scaled_step=True, steps_min=3, steps_max=3
        )
    )
    loop = loop.to(device=device, dtype=dtype)
    trajectory = loop.train_trajectory(model, context, initial)
    loss = cross_entropy(trajectory.candidates[-1], target)
    names = [n for n, _ in model.named_parameters()]
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return names, [g.detach().cpu().double() for g in grads]


def errors(computed: list["torch.Tensor"], reference: list["torch.Tensor"]) -> dict[str, float]:
    """The largest absolute error, the largest elementwise relative error (over elements above
    1e-3 of their tensor's largest reference value) and the largest normwise relative error
    (norm of the error over norm of the reference, per tensor), each the largest over all
    parameter tensors."""
    worst_abs = worst_rel = worst_norm = 0.0
    for c, r in zip(computed, reference, strict=True):
        d = (c - r).abs()
        worst_abs = max(worst_abs, d.max().item())
        significant = r.abs() > 1e-3 * r.abs().max()
        if significant.any():
            worst_rel = max(worst_rel, (d[significant] / r.abs()[significant]).max().item())
        worst_norm = max(worst_norm, (d.norm() / r.norm().clamp_min(1e-300)).item())
    return {"max_abs": worst_abs, "max_rel": worst_rel, "max_normwise_rel": worst_norm}


def measure(device: str = "mps") -> Iterator[dict[str, Any]]:
    """One row per layout, yielded as soon as it is computed: the `errors` of CPU float32 and of
    `device` float32 against CPU float64, and of `device` against CPU float32."""
    import torch

    for layout in LAYOUTS:
        _, reference = gradients("cpu", layout, torch.float64)
        _, cpu32 = gradients("cpu", layout, torch.float32)
        _, device32 = gradients(device, layout, torch.float32)
        yield {
            "layout": layout,
            "cpu_float32": errors(cpu32, reference),
            "mps_float32": errors(device32, reference),
            "mps_vs_cpu_float32": errors(device32, cpu32),
            "reference_max_abs": max(r.abs().max().item() for r in reference),
        }


def round_up(value: float) -> float:
    """`value` rounded up to two significant digits."""
    exponent = math.floor(math.log10(value)) - 1
    return math.ceil(value / 10**exponent) * 10**exponent


@dataclass
class ToleranceDecision:
    """`rounding`: the difference is float32 rounding. `atol`: the absolute tolerance for the
    test, or None if the float32 defaults are kept (or the difference is not rounding)."""

    rounding: bool
    atol: float | None
    within: dict[str, bool] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def decide(rows: list[dict[str, Any]], factor: float = FACTOR) -> ToleranceDecision:
    """The float32-rounding decision and the tolerance.

    The difference is float32 rounding if, in both layouts, the device's largest absolute error
    and its largest normwise relative error are each at most `factor` times those of CPU float32.
    Then the absolute tolerance is (factor + 1) * e, with e the largest CPU float32 absolute error
    over both layouts (the two float32 results can differ by up to the sum of their errors),
    rounded up to two significant digits and at least 1.5 times the largest measured device - CPU
    float32 difference. If it is at most the float32 default of 1e-5, the default tolerances are
    kept."""
    if {row["layout"] for row in rows} != set(LAYOUTS):
        raise ValueError(f"expected one row per layout {LAYOUTS}, got {rows}")
    decision = ToleranceDecision(rounding=True, atol=None)
    for row in rows:
        cpu, mps = row["cpu_float32"], row["mps_float32"]
        within = (
            mps["max_abs"] <= factor * cpu["max_abs"]
            and mps["max_normwise_rel"] <= factor * cpu["max_normwise_rel"]
        )
        decision.within[row["layout"]] = within
        decision.lines.append(
            f"{row['layout']}: CPU32 abs {cpu['max_abs']:.3g} rel {cpu['max_rel']:.3g} "
            f"normwise {cpu['max_normwise_rel']:.3g}; MPS32 abs {mps['max_abs']:.3g} "
            f"rel {mps['max_rel']:.3g} normwise {mps['max_normwise_rel']:.3g}; "
            f"MPS-CPU32 abs {row['mps_vs_cpu_float32']['max_abs']:.3g}; within {within}"
        )
        decision.rounding = decision.rounding and within
    if not decision.rounding:
        decision.lines.append(
            "DECISION: the MPS error is clearly larger than float32 rounding; stop"
        )
        return decision
    cpu_error = max(row["cpu_float32"]["max_abs"] for row in rows)
    measured = max(row["mps_vs_cpu_float32"]["max_abs"] for row in rows)
    atol = round_up(max((factor + 1) * cpu_error, 1.5 * measured))
    if atol <= FLOAT32_DEFAULT_ATOL:
        decision.lines.append(
            f"DECISION: float32 rounding; the derived atol {atol:.2g} is within the float32 "
            "default 1e-5, so the test keeps its default tolerances"
        )
        return decision
    decision.atol = atol
    decision.lines.append(f"DECISION: float32 rounding; atol = {atol:.2g}")
    return decision


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--device", default="mps")
    deciding = commands.add_parser("decide")
    deciding.add_argument("diagnosis", type=Path)
    deciding.add_argument("--factor", type=float, default=FACTOR)
    args = parser.parse_args(argv)
    if args.command == "measure":
        for row in measure(args.device):
            print(json.dumps(row), flush=True)
        return 0
    decision = decide(read_rows(args.diagnosis), args.factor)
    print("\n".join(decision.lines))
    return 0 if decision.rounding else 1


if __name__ == "__main__":
    sys.exit(main())
