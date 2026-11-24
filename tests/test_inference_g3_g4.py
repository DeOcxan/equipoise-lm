"""Questions G3 and G4 on saved checkpoints (src/ebt/experiments/inference_*.py)."""

import math

import pytest
import torch

from ebt.checkpoints import checkpoint_path, save_checkpoint
from ebt.config import ThinkingConfig, Tier
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import inference_mnist as mnist
from ebt.experiments import inference_regression as reg
from ebt.experiments import inference_text as text
from ebt.experiments import mnist_denoising, regression_g2, runner, text_lm
from ebt.fingerprint import code_fingerprint, fingerprint_files
from ebt.results import read_json, write_json
from ebt.tasks.regression import branch_distance, initial_candidates
from ebt.tasks.text import make_data as make_text_data
from support import synthetic_images

CPU = torch.device("cpu")
STEPS = (1, 2, 4)


def test_each_task_fingerprint_covers_its_own_task_code_only():
    names = {
        module: {p.stem for p in fingerprint_files(module)}
        for module in ("inference_g3_g4", *core.TASK_MODULES.values())
    }
    tasks = {"regression_g2", "mnist_denoising", "text_lm", "regression", "mnist", "text"}
    assert not names["inference_g3_g4"] & tasks
    assert names["inference_regression"] & tasks == {"regression_g2", "regression"}
    assert names["inference_mnist"] & tasks == {"mnist_denoising", "mnist"}
    assert names["inference_text"] & tasks == {"text_lm", "text"}
    for name, module in core.TASK_MODULES.items():
        assert "inference_g3_g4" in names[module]
        assert core.load_task(name).module == module and core.load_task(name).name == name


def three_seeds(means, h):
    """Seeds m - h, m, m + h: sample standard deviation exactly h for dyadic h and m."""
    return [[m - h for m in means], list(means), [m + h for m in means]]


def segments(result):
    return [(r["kind"], r["from"], r["to"]) for r in result["ranges"]]


def check_ranges(steps, per_seed, lower_is_better, result):
    """No flat range holds a pair whose change reaches the larger spread; every falling (rising)
    range has falling (rising) ends and no rising (falling) pair."""
    columns = list(zip(*per_seed, strict=True))
    means = [sum(c) / len(c) for c in columns]
    spreads = [core._std(c) for c in columns]

    def kind(i, j):
        change = means[j] - means[i]
        if change == 0 or abs(change) < max(spreads[i], spreads[j]):
            return 0
        return -1 if (change if lower_is_better else -change) < 0 else 1

    sign = {"falling": -1, "flat": 0, "rising": 1}
    for r in result["ranges"]:
        a, b = steps.index(r["from"]), steps.index(r["to"])
        pairs = [kind(i, j) for i in range(a, b + 1) for j in range(i + 1, b + 1)]
        if r["kind"] == "flat":
            assert all(p == 0 for p in pairs), r
        else:
            assert kind(a, b) == sign[r["kind"]] and -sign[r["kind"]] not in pairs, r


def test_ranges_falling_then_flat():
    per_seed = three_seeds([3.0, 2.0, 2.0, 2.0, 2.0], 0.125)
    result = core.classify_ranges((1, 2, 4, 8, 16), per_seed, lower_is_better=True)
    # 3 -> 2 is a fall (1 ≥ 0.125); the rest has zero change: flat from 2 on.
    assert segments(result) == [("falling", 1, 2), ("flat", 2, 16)]
    assert result["rising"] == "absent" and result["undefined"] == "absent"


def test_a_range_with_an_opposite_pair_is_not_monotone():
    # Zero spread, means 2, 1, 3: 1 -> 4 rises, but 1 -> 2 falls, so 1-4 is not a rising range.
    per_seed = [[2.0, 1.0, 3.0], [2.0, 1.0, 3.0]]
    result = core.classify_ranges((1, 2, 4), per_seed, lower_is_better=True)
    assert segments(result) == [("falling", 1, 2), ("rising", 2, 4)]


def test_a_change_before_a_rise_and_a_short_level_pair_inside_a_range():
    # Means 3, 2, 1.9375, 3, sd 0.125: 1->2 falls, 2->4 is level (0.0625), 1->4 falls, 4->8 rises.
    per_seed = three_seeds([3.0, 2.0, 1.9375, 3.0], 0.125)
    result = core.classify_ranges((1, 2, 4, 8), per_seed, lower_is_better=True)
    assert segments(result) == [("falling", 1, 4), ("rising", 4, 8)]
    check_ranges((1, 2, 4, 8), per_seed, True, result)


def test_a_change_equal_to_the_spread_is_not_flat():
    # Change 0.25 against standard deviations 0.125 and 0.25: not smaller than the larger.
    per_seed = [[1.0 - 0.125, 1.25 - 0.25], [1.0, 1.25], [1.0 + 0.125, 1.25 + 0.25]]
    assert segments(core.classify_ranges((1, 2), per_seed, True)) == [("rising", 1, 2)]
    smaller = three_seeds([1.0, 1.1875], 0.25)  # change 0.1875 < 0.25
    assert segments(core.classify_ranges((1, 2), smaller, True)) == [("flat", 1, 2)]


def test_the_spread_is_the_larger_standard_deviation_not_a_pooled_one():
    # Change 0.3125; sd 0.125 and 0.375. Larger: 0.375 -> level. Pooled: √((0.125**2 + 0.375**2)/2)
    # = 0.2795 -> it would rise.
    per_seed = [[1.0 - 0.125, 1.3125 - 0.375], [1.0, 1.3125], [1.0 + 0.125, 1.3125 + 0.375]]
    assert segments(core.classify_ranges((1, 2), per_seed, True)) == [("flat", 1, 2)]


def test_zero_spread_is_flat_only_for_zero_change():
    """Zero change with zero spread is flat and any change with zero spread is not."""
    level = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert segments(core.classify_ranges((1, 2, 4), level, True)) == [("flat", 1, 4)]
    tiny = [[2.0, 2.0 + 2**-20], [2.0, 2.0 + 2**-20]]
    assert segments(core.classify_ranges((1, 2), tiny, True)) == [("rising", 1, 2)]


def test_a_slow_monotone_fall_is_not_all_flat():
    # 0.05 per step with standard deviation 0.1: neighbouring steps are level, but the change
    # over three steps (0.15) is not.
    steps = core.STEP_GRID
    per_seed = three_seeds([1.0 - 0.05 * i for i in range(7)], 0.1)
    result = core.classify_ranges(steps, per_seed, lower_is_better=True)
    check_ranges(steps, per_seed, True, result)
    assert result["ranges"][0]["kind"] == "falling" and result["ranges"][0]["from"] == 1
    assert result["rising"] == "absent"


# Stored per-seed G3 curve of ebt_contrastive (results/regression/g3_g4/full, seeds 0-4).
CONTRASTIVE = [
    [52.6314, 36.0153, 36.2125, 38.3643, 40.3813, 44.2629, 40.7746],
    [77.5625, 59.2676, 64.6568, 66.3077, 60.3383, 67.9996, 59.6146],
    [63.2646, 50.6780, 54.7931, 54.5091, 50.1553, 55.4393, 51.2233],
    [70.1961, 75.3045, 77.4871, 71.1612, 64.6190, 70.9324, 69.3463],
    [64.2400, 56.8067, 58.9742, 56.7428, 57.3930, 56.2560, 56.3911],
]


def test_the_stored_contrastive_curve():
    # Means 65.58, 55.61, 58.42, 57.42, 54.58, 58.98, 55.47; the only pair whose change reaches
    # its spread is 1 -> 16 (-11.00 against 9.53), so steps 1 and 16 are not in one flat range.
    result = core.classify_ranges(core.STEP_GRID, CONTRASTIVE, lower_is_better=True)
    check_ranges(core.STEP_GRID, CONTRASTIVE, True, result)
    assert segments(result) == [("falling", 1, 16), ("flat", 16, 64)]


def test_for_psnr_a_rising_value_is_a_falling_error():
    per_seed = [[20.0, 25.0], [20.0, 25.0]]
    assert core.classify_ranges((1, 2), per_seed, lower_is_better=False)["falling"] == [(1, 2)]
    assert core.classify_ranges((1, 2), per_seed, lower_is_better=True)["rising"] == [(1, 2)]


def test_undefined_steps_form_one_undefined_range():
    per_seed = [[1.0, 2.0, None, None], [1.0, 2.0, 3.0, None]]
    result = core.classify_ranges((1, 2, 4, 8), per_seed, lower_is_better=True)
    assert segments(result) == [("rising", 1, 2), ("undefined", 2, 8)]
    assert result["undefined"] == [(2, 8)]
    nothing = core.classify_ranges((1, 2), [[None, None]], True)
    assert segments(nothing) == [("undefined", 1, 2)]


def test_curve_is_undefined_from_the_first_undefined_step_on():
    values = {k: float(k) for k in core.STEP_GRID}
    assert core.curve(values, None) == [float(k) for k in core.STEP_GRID]
    assert core.curve(values, 4) == [1.0, 2.0, None, None, None, None, None]


@pytest.mark.parametrize(
    ("random", "oracle", "chosen", "expected"),
    [(2.0, 1.0, 1.0, 1.0), (2.0, 1.0, 2.0, 0.0), (2.0, 1.0, 1.5, 0.5), (2.0, 1.0, 3.0, -1.0)],
    ids=["as the oracle", "as random", "half", "worse than random"],
)
def test_selection_fraction(random, oracle, chosen, expected):
    assert core.selection_fraction(random, oracle, chosen) == expected


def test_selection_fraction_is_undefined_when_the_oracle_equals_random():
    assert core.selection_fraction(1.5, 1.5, 1.5) is None
    assert core.selection_fraction(0.0, 0.0, 0.0) is None


def test_the_undefined_rule_is_relative_with_factor_1e_5():
    assert core.UNDEFINED_TOLERANCE == 1e-5
    # Gain 0.5e-5 · random: undefined; 2e-5 · random: defined, at every scale of the error.
    for scale in (1e-3, 1.0, 1e3):
        assert core.selection_fraction(scale, scale * (1 - 0.5e-5), scale) is None
        assert core.selection_fraction(scale, scale * (1 - 2e-5), scale) == pytest.approx(0.0)


def test_best_of_n_by_hand():
    errors = torch.tensor([[3.0, 1.0, 2.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    energies = torch.tensor([[0.5, 0.1, 0.2, 0.9], [0.0, 1.0, 2.0, -1.0]])
    one, two, four = core.best_of_n(errors, energies, (1, 2, 4))
    assert one == {"n": 1, "random": 2.0, "oracle": 2.0, "lowest_energy": 2.0, "fraction": None}
    # N = 2: random (2 + 1.5) / 2, oracle (1 + 1) / 2, lowest energy (1 + 1) / 2.
    assert two == {"n": 2, "random": 1.75, "oracle": 1.0, "lowest_energy": 1.0, "fraction": 1.0}
    # N = 4: random (1.5 + 2.5) / 2 = 2, oracle (0 + 1) / 2, lowest energy (1 + 4) / 2.
    assert four["random"] == 2.0 and four["oracle"] == 0.5 and four["lowest_energy"] == 2.5
    assert four["fraction"] == pytest.approx(-1 / 3)


def test_best_of_n_takes_the_first_of_equal_energies():
    errors = torch.tensor([[5.0, 1.0]])
    energies = torch.tensor([[0.0, 0.0]])
    (row,) = core.best_of_n(errors, energies, (2,))
    assert row["lowest_energy"] == 5.0 and row["fraction"] == -1.0  # (3 - 5) / (3 - 1)


def test_best_of_n_with_identical_candidates_is_undefined():
    errors = torch.ones(3, 4)
    rows = core.best_of_n(errors, torch.randn(3, 4), (1, 2, 4))
    assert [row["fraction"] for row in rows] == [None, None, None]


def test_non_finite_values_never_reach_ranks_or_best_of_n():
    bad = torch.tensor([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError):
        core.ranks(bad)
    with pytest.raises(ValueError):
        core.best_of_n(torch.ones(1, 2), torch.tensor([[0.0, float("inf")]]), (2,))
    assert core.g4_non_finite(bad, None, torch.tensor([float("inf"), 0.0])) == 2


def test_non_finite_g4_value_makes_checkpoint_g4_undefined():
    """A non finite g4 value makes every g4 value of the checkpoint undefined."""
    x = torch.tensor([[4.0], [4.0]])
    candidates = torch.tensor([[2.0, -2.0], [float("nan"), 2.0]])
    energies = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    g4 = reg.regression_g4("ebt_mse", x, candidates, energies)
    assert g4["non_finite_values"] == 2  # the candidate and its error
    assert g4["spearman"] is None
    assert all(row["fraction"] is None and row["random"] is None for row in g4["best_of_n"])
    assert all(row["coverage"] is None for row in g4["coverage"])
    finite = reg.regression_g4("ebt_mse", x, torch.tensor([[2.0, -2.0], [2.0, 2.0]]), energies)
    assert finite["non_finite_values"] == 0 and finite["coverage"][1]["coverage"] == 0.5


def test_both_branch_coverage_by_hand():
    x = torch.tensor([[4.0], [4.0], [4.0], [1.0]])  # branches ±2, ±2, ±2, ±1
    candidates = torch.tensor(
        [
            [2.1, -1.9, 0.0],  # both within 0.2 at N = 2
            [2.0, 2.0, -2.125],  # the negative branch only with the third, r = 0.0625
            [2.3, -2.0, -2.0],  # 2.3 has r = 0.15: positive branch never reached
            [1.0, 0.0, 0.0],  # one branch only
        ]
    )
    rows = reg.both_branch_coverage(x, candidates, (1, 2, 3))
    assert [row["coverage"] for row in rows] == [0.0, 0.25, 0.5]


def test_coverage_uses_the_first_n_candidates():
    x = torch.tensor([[4.0]])
    rows = reg.both_branch_coverage(x, torch.tensor([[2.0, -2.0, 0.0, 0.0]]), (2, 4))
    assert [row["coverage"] for row in rows] == [1.0, 1.0]
    rows = reg.both_branch_coverage(x, torch.tensor([[0.0, 0.0, 2.0, -2.0]]), (2, 4))
    assert [row["coverage"] for row in rows] == [0.0, 1.0]


def test_coverage_includes_r_equal_to_0_1():
    # √6.25 = 2.5; candidates ±2.75 have r = 0.25 / 2.5 = 0.1 exactly in float64.
    x = torch.tensor([[6.25]], dtype=torch.float64)
    candidates = torch.tensor([[2.75, -2.75]], dtype=torch.float64)
    assert reg.both_branch_coverage(x, candidates, (2,))[0]["coverage"] == 1.0


def test_spearman_of_monotone_and_reversed_data():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert core.spearman(a, a.exp()) == pytest.approx(1.0)
    assert core.spearman(a, -(a**3)) == pytest.approx(-1.0)


def test_spearman_with_ties_by_hand():
    # Ranks of a: 1, 2.5, 2.5, 4; of b: 1, 2, 3, 4. Centred: (-1.5, 0, 0, 1.5) and
    # (-1.5, -0.5, 0.5, 1.5); ρ = 4.5 / sqrt(4.5 · 5) = 0.9487.
    a = torch.tensor([1.0, 2.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert core.ranks(a).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert core.spearman(a, b) == pytest.approx(4.5 / math.sqrt(4.5 * 5))


def test_spearman_of_a_constant_is_undefined():
    assert core.spearman(torch.ones(4), torch.arange(4.0)) is None


def test_evaluation_seeds_overlap_no_training_or_fixed_seed():
    used = set()
    for experiment in (regression_g2, mnist_denoising, text_lm):
        used |= set(experiment.FIXED_SEEDS.values())
        for run in (experiment.SEARCH_SEED, *experiment.REPORT_SEEDS):
            used |= set(experiment.run_seeds(run).values())
    evaluation = [
        value
        for seed in regression_g2.REPORT_SEEDS
        for value in core.evaluation_seeds(seed).values()
    ]
    assert len(evaluation) == len(set(evaluation))
    assert not used & set(evaluation)
    assert min(evaluation) > max(used)


def test_interval_by_hand():
    value = core.interval([1.0, 2.0, 3.0, None, float("nan")])
    assert value["defined"] == 3 and value["mean"] == 2.0 and value["std"] == 1.0
    # t(0.975, 2) = 4.3027 times 1 / √3.
    assert value["half_width"] == pytest.approx(4.302653 / math.sqrt(3), rel=1e-5)
    assert core.interval([None, None]) == {
        "values": [None, None],
        "defined": 0,
        "mean": None,
        "half_width": None,
    }
    single = core.interval([0.5, None])
    assert single["mean"] == 0.5 and single["half_width"] is None and "std" not in single


def fake_result(g3, spearman, bon2, coverage):
    """A result file of one seed with the fields summarize reads."""
    return {
        "g3": {"branch_distance": g3, "on_branch": [0.0] * 7},
        "g4": {
            "non_finite_values": 0,
            "spearman": spearman,
            "best_of_n": [
                {"n": 1, "random": 1.0, "oracle": 1.0, "lowest_energy": 1.0, "fraction": None},
                {"n": 2, **bon2},
            ],
            "coverage": [{"n": 1, "coverage": coverage}],
        },
    }


def summary_inputs(tmp_path):
    seed0 = fake_result(
        [3.0] + [2.0] * 6,
        0.5,
        {"random": 2.0, "oracle": 1.0, "lowest_energy": 1.0, "fraction": 1.0},
        0.0,
    )
    bad = {"random": 10.0, "oracle": 0.0, "lowest_energy": 10.0, "fraction": 0.0}
    seed1 = fake_result([3.0] + [2.0] * 6, None, bad, 0.5)
    write_json(tmp_path / "m-seed0.json", seed0)
    write_json(tmp_path / "m-seed1.json", seed1)
    good = {"random": 2.0, "oracle": 1.0, "lowest_energy": 1.0, "fraction": 1.0}
    gone = fake_result([1.0, 1.0, 1.0, 1.0, None, None, None], 0.1, good, 0.0)
    write_json(tmp_path / "u-seed0.json", gone)
    write_json(tmp_path / "u-seed1.json", gone)
    return core.summarize(reg.TASK, ["m", "u"], [0, 1], tmp_path)


def test_summarize_averages_per_seed_values(tmp_path):
    m = summary_inputs(tmp_path)["models"]["m"]
    # The fraction is the mean of the seeds' fractions (1 and 0), not the fraction of the mean
    # errors ((6 - 5.5) / (6 - 0.5) = 0.09).
    two = m["g4"]["best_of_n"][1]
    assert two["fraction"]["mean"] == 0.5
    assert two["random"]["mean"] == 6.0 and two["oracle"]["mean"] == 0.5
    assert m["g4"]["spearman"]["defined"] == 1 and m["g4"]["spearman"]["mean"] == 0.5
    coverage = m["g4"]["coverage"][0]["coverage"]
    # Mean 0.25; t(0.975, 1) = 12.706 times sd 0.3536 / √2.
    assert coverage["mean"] == 0.25
    assert coverage["half_width"] == pytest.approx(12.7062 * 0.25, rel=1e-4)
    # Identical seeds, 3 then 2 everywhere: zero spread, falling 1-2 and zero change after it.
    assert segments(m["g3"]["classification"]) == [("falling", 1, 2), ("flat", 2, 64)]
    assert m["g3"]["per_step"][0] == {
        "values": [3.0, 3.0],
        "defined": 2,
        "mean": 3.0,
        "half_width": 0.0,
        "std": 0.0,
    }


def test_table_prints_ranges_undefined_values_and_counts(tmp_path):
    summary = summary_inputs(tmp_path)
    text_table = core.table("regression", summary)
    assert "Ranges: falling 1–2; flat 2–64; rising absent; undefined absent." in text_table
    assert "Ranges: falling absent; flat 1–8; rising absent; undefined 8–64." in text_table
    assert "In order: falling 1–2 → flat 2–64." in text_table
    assert "| 16 | undefined |  |" in text_table
    assert "- Spearman ρ(final energy, error): 0.5000 (1 of 2 defined)" in text_table
    assert "| 2 | 6.0000 ± 50.8248 | 5.5000 ± 57.1779 | 0.5000 ± 6.3531 | 0.500 ± 6.353 |" in (
        text_table
    )


def regression_checkpoint(directory, model, seed=0):
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    result, _, trained = regression_g2.run(candidate, Tier.FAST, seed, CPU, with_test=True)
    path = checkpoint_path(directory, model, seed)
    save_checkpoint(path, trained, code_fingerprint("regression_g2"), result)
    return path, result


def test_single_steps_equal_one_call_with_many_steps(tmp_path):
    path, _ = regression_checkpoint(tmp_path, "ebt_mse")
    setup = regression_g2.restore(path, CPU)
    model, loop = setup.objective.model, setup.objective.loop
    x = torch.linspace(0.5, 3.5, 16).unsqueeze(-1)
    initial = torch.randn(16, 1, generator=torch.Generator().manual_seed(3))
    recorded = {}
    core.think(model, loop, x, initial, STEPS, lambda k, c: recorded.update({k: c.clone()}))
    for k in STEPS:
        config = regression_g2.INFERENCE.__class__(steps=k)
        assert torch.equal(recorded[k], loop.infer(model, x, initial, config).candidates[-1])


def test_fixed_step_loop_reproduces_trained_loop(tmp_path):
    """A fixed step loop with the trained step size reproduces the trained loop."""
    path, _ = regression_checkpoint(tmp_path, "ebt_regularized")
    setup = regression_g2.restore(path, CPU)
    model, trained = setup.objective.model, setup.objective.loop
    fresh = reg.fixed_step_loop(trained.step_size.raw.item(), CPU)
    x = torch.linspace(0.5, 3.5, 16).unsqueeze(-1)
    initial = torch.randn(16, 1, generator=torch.Generator().manual_seed(4))
    config = regression_g2.INFERENCE
    same = trained.infer(model, x, initial, config).candidates[-1]
    assert torch.equal(fresh.infer(model, x, initial, config).candidates[-1], same)


def test_thinking_reports_the_first_non_finite_step():
    class Exploding(regression_g2.MLPEnergy):
        def energy(self, context, encoded):
            return (encoded.squeeze(-1) * 1e30) ** 3

    model = Exploding(1, 1, regression_g2.ENERGY_MODEL)
    loop = ThinkingLoop(ThinkingConfig(step_size=1.0))
    recorded = []
    stop = core.think(
        model, loop, torch.ones(2, 1), torch.ones(2, 1), STEPS, lambda k, c: recorded.append(k)
    )
    assert stop == 1 and recorded == []


def test_the_accept_only_lower_step_never_raises_the_energy(tmp_path):
    path, _ = regression_checkpoint(tmp_path, "ebt_regularized")
    setup = regression_g2.restore(path, CPU)
    model = setup.objective.model
    loop = reg.fixed_step_loop(50.0, CPU)  # far too long a step: most proposals are worse
    x = torch.linspace(0.5, 3.5, 64).unsqueeze(-1)
    initial = torch.randn(64, 1, generator=torch.Generator().manual_seed(5))
    energies = []

    def record(k, candidate):
        energies.append(reg.final_energy(model, x, candidate))

    core.think(model, loop, x, initial, (1, 2, 3, 4), record, reg.accept_lower)
    start = reg.final_energy(model, x, initial)
    for before, after in zip([start, *energies], energies, strict=False):
        assert (after <= before).all()


def evaluation_seed_of_run(monkeypatch, seed_name, value):
    original = core.evaluation_seeds
    monkeypatch.setattr(core, "evaluation_seeds", lambda seed: {**original(seed), seed_name: value})


def test_regression_g3_at_trained_steps_equals_trained_inference(tmp_path):
    """Regression g3 at the trained step count equals the trained inference."""
    path, result = regression_checkpoint(tmp_path, "ebt_regularized")
    evaluated = reg.evaluate(path, CPU, None, None)
    # The trained inference (4 steps) from the same initial candidates, as the final run does it.
    setup = regression_g2.restore(path, CPU)
    data = regression_g2.make_data(regression_g2.TIERS[Tier.FAST].data, regression_g2.DATA_SEED)
    seed = core.evaluation_seeds(0)["g3_initial"]
    predictions = setup.predict(data.test, torch.Generator().manual_seed(seed))
    expected = regression_g2.regression_metrics(data.test, predictions.final, None)
    at_four = evaluated["g3"]["branch_distance"][core.STEP_GRID.index(4)]
    assert at_four == expected["branch_distance"]
    plain_full_step = next(
        row for row in evaluated["c3"] if row["divisor"] == 1 and not row["accept_lower"]
    )
    assert plain_full_step["branch_distance"] == evaluated["g3"]["branch_distance"]
    assert len(evaluated["c3"]) == 2 * len(reg.C3_DIVISORS)
    trained = result.config["thinking"]["step_size"]  # α in float32 in the checkpoint
    for row in evaluated["c3"]:
        assert row["step_size"] == pytest.approx(trained / row["divisor"], rel=1e-6)
    # One plain step of α / 16 from the same initial candidates, computed here directly.
    model = setup.objective.model
    initial = initial_candidates(
        data.test,
        regression_g2.make_data(regression_g2.TIERS[Tier.FAST].data, 0).init_std,
        torch.Generator().manual_seed(seed),
    )
    x = data.test.context
    with torch.enable_grad():
        point = initial.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(model.energy(x, point).sum(), point)
    alpha = setup.objective.loop.step_size.raw.item() / 16
    one_step = (initial - alpha * gradient).detach()
    expected_r = branch_distance(x, one_step).mean().item()
    sixteenth = next(r for r in evaluated["c3"] if r["divisor"] == 16 and not r["accept_lower"])
    assert sixteenth["branch_distance"][0] == pytest.approx(expected_r, rel=1e-6)
    # Accept-only-lower steps at the trained α: the curve equals the one computed here with that
    # step, and it differs from the plain curve.
    loop = reg.fixed_step_loop(setup.objective.loop.step_size.raw.item(), CPU)
    init_std = regression_g2.make_data(regression_g2.TIERS[Tier.FAST].data, 0).init_std
    lower = reg.branch_curve(model, loop, data.test, init_std, 0, CPU, reg.accept_lower)
    accepted = next(r for r in evaluated["c3"] if r["divisor"] == 1 and r["accept_lower"])
    assert accepted["branch_distance"] == lower["branch_distance"]
    assert accepted["branch_distance"] != plain_full_step["branch_distance"]


def test_regression_g4_candidates_come_from_the_g4_seed(tmp_path):
    path, _ = regression_checkpoint(tmp_path, "ebt_mse")
    evaluated = reg.evaluate(path, CPU, None, 64)
    setup = regression_g2.restore(path, CPU)
    model, loop = setup.objective.model, setup.objective.loop
    data = regression_g2.make_data(regression_g2.TIERS[Tier.FAST].data, regression_g2.DATA_SEED)
    test = data.test.rows(slice(0, 64))

    def first_candidate_error(source):
        generator = torch.Generator().manual_seed(core.evaluation_seeds(0)[source])
        initial = initial_candidates(test, data.init_std, generator)
        final = loop.infer(model, test.context, initial, regression_g2.INFERENCE).candidates[-1]
        return branch_distance(test.context, final).mean().item()

    first = evaluated["g4"]["best_of_n"][0]["random"]
    assert first == pytest.approx(first_candidate_error("g4_candidates"), rel=1e-6)
    assert first != pytest.approx(first_candidate_error("g3_initial"), rel=1e-3)


@pytest.mark.parametrize("model", ["mdn", "ired", "ibc"])
def test_regression_g4_for_the_models_without_thinking_steps(tmp_path, model):
    path, _ = regression_checkpoint(tmp_path, model)
    evaluated = reg.evaluate(path, CPU, None, 32)
    assert "g3" not in evaluated
    g4 = evaluated["g4"]
    assert g4["non_finite_values"] == 0
    assert [row["n"] for row in g4["coverage"]] == list(core.CANDIDATE_GRID)
    assert all(0 <= row["coverage"] <= 1 for row in g4["coverage"])
    assert ("best_of_n" in g4) == (model != "mdn")


def test_mnist_evaluation_reproduces_the_final_run_at_two_steps(tmp_path):
    data = synthetic_images()
    candidate = mnist_denoising.search_space("ebt", Tier.FAST)[0]
    result, _, trained = mnist_denoising.run(candidate, Tier.FAST, 0, CPU, True, data)
    path = checkpoint_path(tmp_path, "ebt", 0)
    save_checkpoint(path, trained, code_fingerprint("mnist_denoising"), result)
    evaluated = mnist.evaluate(path, CPU, data, None)
    test_psnr = result.metrics["test"]["psnr"]
    assert evaluated["g3"]["psnr"][core.STEP_GRID.index(2)] == pytest.approx(test_psnr, rel=1e-9)
    g4 = evaluated["g4"]
    assert g4["non_finite_values"] == 0
    assert g4["standard_psnr"] == pytest.approx(test_psnr, rel=1e-9)
    assert g4["perturbation_std"] == pytest.approx(0.3209, abs=1e-4)
    assert g4["best_of_n"][0]["lowest_energy_psnr"] == pytest.approx(g4["perturbed_first_psnr"])
    assert g4["perturbed_first_psnr"] != g4["standard_psnr"]
    assert -1 <= g4["spearman"] <= 1


TEXT_CORPUS = "".join(f"Line {i % 7}: to be, or not to be; that is {i % 3}.\n" for i in range(300))
TEXT_OOD = (
    "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
    "Chapter I.]\n" + "It is a truth universally acknowledged. " * 500 + "\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
)


def test_text_evaluation_on_the_fast_tier_reproduces_the_final_run(tmp_path, monkeypatch):
    data = make_text_data(TEXT_CORPUS, TEXT_OOD, text_lm.TIERS[Tier.FAST].sequence_length)
    candidate = text_lm.search_space("ebt", Tier.FAST, tmp_path)[0]
    result, _, trained = text_lm.run(candidate, Tier.FAST, 0, CPU, True, data)
    path = checkpoint_path(tmp_path, "ebt", 0)
    save_checkpoint(path, trained, code_fingerprint("text_lm"), result)
    # With the initial logits of the final run's test evaluation, 4 steps give its test BPC.
    evaluation_seed_of_run(monkeypatch, "g3_initial", text_lm.FIXED_SEEDS["test_initial"])
    evaluated = text.evaluate(path, CPU, data, None)
    at_four = evaluated["g3"]["bpc"][core.STEP_GRID.index(4)]
    assert at_four == pytest.approx(result.metrics["test"]["bpc"], rel=1e-9)
    g4 = evaluated["g4"]
    assert g4["non_finite_values"] == 0
    assert g4["spearman_pairs"] == data.test.target.numel()  # one pair per position
    assert [row["n"] for row in g4["best_of_n"]] == list(core.CANDIDATE_GRID)
    assert g4["best_of_n"][0]["fraction"] is None


def test_results_are_written_skipped_and_stale_ones_refused(tmp_path):
    checkpoints, output = tmp_path / "checkpoints", tmp_path / "out"
    for seed in (0, 1):
        regression_checkpoint(checkpoints, "mdn", seed)
    common = (reg.TASK, ["mdn"], [0, 1], checkpoints, output, CPU, None)
    summary = core.execute(*common, reuse_stale=False, rows=16)
    first = output / "mdn-seed0.json"
    stored = read_json(first)
    assert stored["code_fingerprint"] == code_fingerprint("inference_regression")
    assert stored["fingerprint_module"] == "inference_regression"
    assert stored["checkpoint"]["code_fingerprint"] == code_fingerprint("regression_g2")
    assert len(stored["checkpoint"]["sha256"]) == 64
    assert summary["models"]["mdn"]["g4"]["coverage"][0]["coverage"]["defined"] == 2

    before = first.read_bytes()
    core.execute(*common, reuse_stale=False, rows=16)
    assert first.read_bytes() == before  # skipped

    write_json(first, {**stored, "code_fingerprint": "other code"})
    with pytest.raises(runner.StaleResultsError):
        core.execute(*common, reuse_stale=False, rows=16)
    summary = core.execute(*common, reuse_stale=True, rows=16)
    assert summary["reused_stale_files"] == [str(first)]
    assert read_json(first)["code_fingerprint"] == "other code"  # kept, not recomputed


def test_a_result_from_another_checkpoint_or_setting_is_stale(tmp_path):
    checkpoints, output = tmp_path / "checkpoints", tmp_path / "out"
    regression_checkpoint(checkpoints, "mdn", 0)
    common = (reg.TASK, ["mdn"], [0], checkpoints, output, CPU, None)
    core.execute(*common, reuse_stale=False, rows=16)
    with pytest.raises(runner.StaleResultsError):
        core.execute(*common, reuse_stale=False, rows=32)
    regression_checkpoint(checkpoints, "mdn", 0)  # retrained: new file, new SHA-256
    stored = read_json(output / "mdn-seed0.json")
    write_json(
        output / "mdn-seed0.json",
        {**stored, "checkpoint": {**stored["checkpoint"], "sha256": "0" * 64}},
    )
    with pytest.raises(runner.StaleResultsError):
        core.execute(*common, reuse_stale=False, rows=16)


def test_a_checkpoint_from_other_code_is_refused_or_marked(tmp_path):
    checkpoints, output = tmp_path / "checkpoints", tmp_path / "out"
    candidate = regression_g2.search_space("mdn", Tier.FAST)[0]
    result, _, trained = regression_g2.run(candidate, Tier.FAST, 0, CPU, with_test=False)
    save_checkpoint(checkpoint_path(checkpoints, "mdn", 0), trained, "old code", result)
    with pytest.raises(runner.StaleResultsError):
        core.execute(reg.TASK, ["mdn"], [0], checkpoints, output, CPU, None, False, 16)
    assert not output.exists()
    core.execute(reg.TASK, ["mdn"], [0], checkpoints, output, CPU, None, True, 16)
    assert read_json(output / "mdn-seed0.json")["checkpoint_from_other_code"] is True


def test_a_missing_checkpoint_stops_before_any_work(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.execute(reg.TASK, ["mdn"], [0], tmp_path, tmp_path / "out", CPU, None, False)
    assert not (tmp_path / "out").exists()
