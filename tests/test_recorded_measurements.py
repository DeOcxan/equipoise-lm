"""The budget decision of method.md 18.8 and the MPS tolerance of docs/numerics.md follow from the
recorded measurements in results/text/."""

import json
from pathlib import Path

import pytest

from tools import mps_text_gradients, text_budget

RECORDED = Path(__file__).resolve().parents[1] / "results/text/measurements-2026-11-21"
"""The measurements of the model in use, from which the setting of the text task was chosen."""


def test_budget_rule_on_recorded_measurements_gives_recorded_choice():
    """The rule applied to the recorded measurements gives the recorded choice.json and the
    recorded section-18.8.md, character for character."""
    budget = RECORDED / "budget"
    records = {
        text_budget.record_key(record): record
        for record in text_budget.read_lines(budget / "measurements.jsonl")
    }
    failed = text_budget.read_failed(budget / "oom.jsonl")
    memory = json.loads((budget / "memory.json").read_text())
    decision = text_budget.decide(records, failed, memory["free_gib"])
    assert decision.missing is None
    assert decision.choice == json.loads((budget / "choice.json").read_text())
    assert decision.choice["dim"] == 256 and decision.choice["layout"] == "split"
    section = text_budget.section(memory, decision, records, failed)
    assert section == (budget / "section-18.8.md").read_text()


def test_the_mps_decision_on_the_recorded_diagnosis_gives_atol_7_2e_05():
    folder = RECORDED / "mps-gradients"
    rows = mps_text_gradients.read_rows(folder / "mps-diag.jsonl")
    decision = mps_text_gradients.decide(rows, mps_text_gradients.FACTOR)
    assert decision.rounding
    assert decision.atol == pytest.approx(7.2e-5)
    assert "\n".join(decision.lines) + "\n" == (folder / "mps-decision.txt").read_text()


FIRST = Path(__file__).resolve().parents[1] / "results/text/measurements-2026-11-20"
"""The first measurements, of the model before corrections 13.1 to 13.4 (method.md 18.8)."""


def first_records():
    lines = (FIRST / "measurements.jsonl").read_text().splitlines()
    return {text_budget.record_key(r): r for r in map(json.loads, lines)}


def test_first_measurements_give_128x4_projections():
    """The first measurements give the projected hours of 128 x 4, S 256, batch 64, split layout,
    for the search and the final runs at T = 1 000, 2 000 and 5 000."""
    records = first_records()
    assert len(records) == 23
    baseline = records[text_budget.key(128, 4, 256, 64, "transformer", "split")]
    ebt = records[text_budget.key(128, 4, 256, 64, "ebt", "split")]
    expected = {1000: (1.55, 1.95), 2000: (3.07, 3.85), 5000: (7.63, 9.55)}
    for length, (search, finals) in expected.items():
        hours = text_budget.projection(baseline, ebt, length)
        assert hours == pytest.approx((search, finals), abs=0.005)


def test_first_measurements_give_application_of_item_3():
    """The rule of method.md 18.7 on the first measurements chooses 256 x 6 at S 256 and batch 32
    and T = 2 000. 384 x 6 was not measured and is given as out of memory; 256 x 6 at S 256 and
    batch 64 is out of memory in the full layout (oom.jsonl)."""
    failed = text_budget.read_failed(FIRST / "oom.jsonl")
    assert failed == {text_budget.key(256, 6, 256, 64, "ebt", "full"): text_budget.OUT_OF_MEMORY}
    failed |= {
        text_budget.key(384, 6, s, b, "ebt", "split"): text_budget.OUT_OF_MEMORY
        for s, b in text_budget.SETTINGS
    }
    decision = text_budget.decide(first_records(), failed, 0.64 * 16)
    choice = decision.choice
    assert (choice["dim"], choice["sequence_length"], choice["batch_size"]) == (256, 256, 32)
    assert choice["train_steps"] == 2000
    assert (choice["search_hours"], choice["final_hours"]) == pytest.approx((5.08, 6.39), abs=0.005)
