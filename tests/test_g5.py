"""Question G5 (method.md 20): src/ebt/experiments/inference_g5.py and tools/g5_report.py."""

import math

import pytest
import torch

from ebt.checkpoints import checkpoint_path, save_checkpoint
from ebt.config import Tier
from ebt.experiments import inference_g3_g4 as core
from ebt.experiments import inference_g5 as g5
from ebt.experiments import (
    inference_mnist,
    inference_regression,
    inference_text,
    mnist_denoising,
    regression_g2,
    runner,
    text_lm,
)
from ebt.fingerprint import code_fingerprint, fingerprint_files
from ebt.results import read_json, write_json
from ebt.statistics import student_t_quantile
from ebt.tasks.mnist import alpha_bar
from ebt.tasks.text import make_data as make_text_data
from support import synthetic_images
from tools import g5_report as report

CPU = torch.device("cpu")
FAST_REGRESSION = regression_g2.TIERS[Tier.FAST].data


def test_the_regression_improvement_is_the_relative_reduction_of_r():
    assert report.improvement("branch_distance", 0.5, 0.25) == pytest.approx(0.5)
    assert report.improvement("branch_distance", 0.4, 0.6) == pytest.approx(-0.5)
    assert report.improvement("branch_distance", 0.3, 0.3) == 0


def test_the_text_improvement_is_taken_on_the_perplexity():
    # BPC 2 -> 1: perplexity 4 -> 2, so I = 1 - 2 / 4 = 0.5 (not 1 - 1 / 2 on the BPC itself).
    assert report.improvement("bpc", 2.0, 1.0) == pytest.approx(0.5)
    assert report.error_of("bpc", 3.0) == pytest.approx(8.0)
    # BPC 2.5 -> 2.3: I = 1 - 2^(-0.2).
    assert report.improvement("bpc", 2.5, 2.3) == pytest.approx(1 - 2**-0.2)


def test_the_mnist_improvement_is_taken_on_the_mse_behind_the_psnr():
    # PSNR 20 dB -> 30 dB: MSE 255**2 / 100 -> 255**2 / 1000, so I = 1 - 1 / 10 = 0.9.
    assert report.error_of("psnr", 20.0) == pytest.approx(255**2 / 100)
    assert report.improvement("psnr", 20.0, 30.0) == pytest.approx(0.9)
    # A falling PSNR is a negative improvement: 30 -> 27 dB, I = 1 - 10^0.3.
    assert report.improvement("psnr", 30.0, 27.0) == pytest.approx(1 - 10**0.3)


def test_an_undefined_or_zero_first_error_gives_an_undefined_improvement():
    assert report.improvement("branch_distance", 0.0, 0.1) is None
    assert report.improvement("branch_distance", None, 0.1) is None
    assert report.improvement("psnr", 20.0, None) is None
    assert report.improvement("bpc", math.inf, 1.0) is None


def test_the_improvement_uses_step_1_and_step_64_of_the_curve():
    values = [0.8, 0.7, 0.6, 0.5, 0.45, 0.42, 0.4]
    g3 = {"steps": list(core.STEP_GRID), "branch_distance": values}
    assert report.curve_improvement("branch_distance", g3) == pytest.approx(1 - 0.4 / 0.8)


def test_delta_per_seed_and_its_interval_by_hand():
    inside, outside = [0.1, 0.2, 0.3, None], [0.1, 0.3, 0.5, 0.4]
    delta = report.differences(inside, outside)
    assert delta[:3] == pytest.approx([0.0, 0.1, 0.2]) and delta[3] is None
    summary = core.interval(delta)
    # Mean 0.1, sample standard deviation 0.1, n = 3: half-width t(0.975, 2) · 0.1 / √3.
    assert summary["defined"] == 3
    assert summary["mean"] == pytest.approx(0.1)
    assert summary["half_width"] == pytest.approx(4.302652729749464 * 0.1 / math.sqrt(3))
    assert student_t_quantile(0.975, 2) == pytest.approx(4.302652729749464)


def test_an_undefined_value_at_64_steps_makes_i_undefined():
    # No fallback to an earlier step: with 64 undefined, I is undefined even if 4 is defined.
    values = [0.8, 0.7, 0.6, 0.5, None, None, None]
    g3 = {"steps": list(core.STEP_GRID), "branch_distance": values}
    assert report.curve_improvement("branch_distance", g3) is None


@pytest.mark.parametrize(
    ("mean", "half", "is_above", "is_below"),
    [
        (0.2, 0.1, True, False),
        (0.1, 0.1, False, False),  # lower end exactly 0
        (math.nextafter(0.1, 1.0), 0.1, True, False),
        (-0.1, 0.1, False, False),  # upper end exactly 0
        (math.nextafter(-0.1, -1.0), 0.1, False, True),
        (0.0, 0.0, False, False),
        (0.3, None, False, False),
        (None, None, False, False),
    ],
)
def test_an_interval_is_above_or_below_0_only_if_it_excludes_0(mean, half, is_above, is_below):
    summary = {"mean": mean, "half_width": half}
    assert report.above(summary) is is_above
    assert report.below(summary) is is_below


def s(mean, half):
    return {"mean": mean, "half_width": half}


@pytest.mark.parametrize(
    ("i_id", "i_ood", "delta", "expected"),
    [
        (s(0.1, 0.05), s(0.3, 0.1), s(0.2, 0.1), "helps_more"),
        # The Δ interval touches 0: not more, and nothing else applies.
        (s(0.1, 0.05), s(0.2, 0.1), s(0.1, 0.1), "no_difference"),
        # Both negative and Δ above 0: a smaller degradation, not "helps more".
        (s(-0.5, 0.1), s(-0.2, 0.05), s(0.3, 0.1), "degrades_both_smaller"),
        (s(-0.5, 0.1), s(-0.1, 0.05), s(0.4, 0.1), "degrades_both_smaller"),
        (s(-0.2, 0.1), s(-0.5, 0.1), s(-0.3, 0.1), "degrades_both_larger"),
        (s(-0.2, 0.1), s(-0.25, 0.1), s(-0.05, 0.1), "degrades_both_same"),
        # I_OOD negative only on average (its interval includes 0): still both negative.
        (s(-0.3, 0.1), s(-0.05, 0.2), s(0.25, 0.1), "degrades_both_smaller"),
        (s(0.1, 0.05), s(-0.2, 0.1), s(-0.3, 0.1), "degrades_ood_only"),
        (s(0.05, 0.1), s(0.15, 0.2), s(0.1, 0.05), "difference_only"),
        (s(0.3, 0.1), s(0.1, 0.05), s(-0.2, 0.1), "helps_less"),
        (s(0.2, 0.1), s(0.2, 0.1), s(0.0, 0.1), "no_difference"),
        # The mean of I_ID below 0 with its interval including 0: still outcome 3.
        (s(-0.1, 0.2), s(-0.3, 0.1), s(-0.2, 0.1), "degrades_both_larger"),
        # The mean of I_ID exactly 0 is not below 0.
        (s(0.0, 0.1), s(-0.3, 0.1), s(-0.3, 0.1), "degrades_ood_only"),
        # The mean of I_OOD below 0 with its interval including 0: not outcome 4.
        (s(0.3, 0.1), s(-0.05, 0.2), s(-0.35, 0.1), "helps_less"),
        # Outcome 2 with the mean of I_ID below 0.
        (s(-0.1, 0.2), s(0.3, 0.1), s(0.4, 0.1), "helps_more"),
        (s(0.2, None), s(0.2, 0.1), s(0.0, 0.1), "undefined"),
        (s(0.2, 0.1), s(None, None), s(None, None), "undefined"),
    ],
)
def test_the_outcomes_of_the_decision_rule(i_id, i_ood, delta, expected):
    assert report.outcome(i_id, i_ood, delta) == expected
    assert expected in report.OUTCOMES or expected in report.DEGRADATION


def test_no_basic_outcome_text_refers_to_the_paper():
    for key, text in report.OUTCOMES.items():
        assert "paper" not in text, key


HELPS_MORE = "Thinking helps more out of distribution."
PAPER = " This is the effect the paper reports for language (§4.1.3)."
NOT_PAPER = " This is not the paper's effect."


@pytest.mark.parametrize(
    ("task", "expected"),
    [("text", HELPS_MORE + PAPER), ("mnist", HELPS_MORE), ("regression", HELPS_MORE)],
)
def test_outcome_2_names_the_papers_effect_for_the_text_task_only(task, expected):
    text = report.outcome_text("helps_more", task, s(0.1, 0.05), s(0.3, 0.1))
    assert text == expected
    assert ("paper" in text) is (task == "text")


@pytest.mark.parametrize("task", ["text", "mnist", "regression"])
def test_outcome_2_negative_i_id_states_in_distribution_degradation(task):
    """Outcome 2 with a negative mean of i id states the in distribution degradation."""
    i_id, i_ood, delta = s(-0.1, 0.2), s(0.3, 0.1), s(0.4, 0.1)
    assert report.outcome(i_id, i_ood, delta) == "helps_more"
    expected = (
        "Thinking helps out of distribution and degrades the result in distribution; Δ "
        "therefore includes the in-distribution degradation."
    )
    assert report.outcome_text("helps_more", task, i_id, i_ood) == expected + (
        PAPER_OOD if task == "text" else ""
    )


PAPER_OOD = " Out of distribution, this matches the effect the paper reports for language (§4.1.3)."


def test_outcome_2_text_negative_i_id_matches_paper_ood():
    """Outcome 2 for text with a negative mean of i id matches the paper out of distribution."""
    i_id, i_ood = s(-0.1, 0.2), s(0.3, 0.1)
    assert report.outcome_text("helps_more", "text", i_id, i_ood) == (
        "Thinking helps out of distribution and degrades the result in distribution; Δ therefore "
        "includes the in-distribution degradation. Out of distribution, this matches the effect "
        "the paper reports for language (§4.1.3)."
    )
    assert PAPER not in report.outcome_text("helps_more", "text", i_id, i_ood)


@pytest.mark.parametrize("task", ["mnist", "regression"])
def test_outcome_2_negative_i_id_no_paper_sentence_outside_text(task):
    """Outcome 2 with a negative mean of i id has no paper sentence outside text."""
    text = report.outcome_text("helps_more", task, s(-0.1, 0.2), s(0.3, 0.1))
    assert PAPER_OOD not in text and PAPER not in text and "paper" not in text


@pytest.mark.parametrize("task", ["text", "mnist", "regression"])
def test_outcome_6_negative_i_ood_does_not_say_thinking_helps(task):
    """Outcome 6 with a negative mean of i ood does not say thinking helps."""
    i_id, i_ood, delta = s(0.3, 0.1), s(-0.05, 0.2), s(-0.35, 0.1)
    assert report.outcome(i_id, i_ood, delta) == "helps_less"
    text = report.outcome_text("helps_less", task, i_id, i_ood)
    assert text == (
        "Thinking does not help out of distribution on average; its effect there is smaller "
        "than in distribution."
    )
    assert not text.startswith("Thinking helps")
    positive = report.outcome_text("helps_less", task, s(0.3, 0.1), s(0.1, 0.05))
    assert positive == "Thinking helps less out of distribution."


@pytest.mark.parametrize(
    ("i_id", "i_ood", "delta", "expected"),
    [
        (
            s(-0.5, 0.1),
            s(-0.2, 0.05),
            s(0.3, 0.1),
            "Thinking degrades the result in both; the relative degradation out of distribution "
            "is smaller.",
        ),
        (
            s(-0.3, 0.1),
            s(-0.05, 0.2),
            s(0.25, 0.1),
            "Thinking degrades the result in distribution and out of distribution on average; the "
            "relative degradation out of distribution is smaller.",
        ),
        (
            s(-0.1, 0.2),
            s(-0.3, 0.1),
            s(-0.2, 0.1),
            "Thinking degrades the result in distribution on average and out of distribution; the "
            "relative degradation out of distribution is larger.",
        ),
        (
            s(-0.1, 0.2),
            s(-0.1, 0.25),
            s(0.0, 0.1),
            "Thinking degrades the result in distribution on average and out of distribution on "
            "average; the relative degradation out of distribution is not measurably different.",
        ),
        (
            s(-0.2, 0.1),
            s(-0.5, 0.1),
            s(-0.3, 0.1),
            "Thinking degrades the result in both; the relative degradation out of distribution "
            "is larger.",
        ),
    ],
)
def test_outcome_3_says_on_average_for_a_side_decided_on_its_mean_only(
    i_id, i_ood, delta, expected
):
    key = report.outcome(i_id, i_ood, delta)
    assert key.startswith("degrades_both")
    assert report.outcome_text(key, "text", i_id, i_ood) == expected + NOT_PAPER
    for task in ("mnist", "regression"):
        assert report.outcome_text(key, task, i_id, i_ood) == expected


DEGRADES_OOD_ONLY = "Thinking degrades the result out of distribution only."
DIFFERENCE_ONLY = (
    "The difference is above 0 but the out-of-distribution improvement is not measurably above 0."
)
OUTCOMES_3_TO_5 = [
    (
        s(-0.5, 0.1),
        s(-0.2, 0.05),
        s(0.3, 0.1),
        "degrades_both_smaller",
        "Thinking degrades the result in both; the relative degradation out of distribution is "
        "smaller.",
    ),
    (
        s(-0.2, 0.1),
        s(-0.5, 0.1),
        s(-0.3, 0.1),
        "degrades_both_larger",
        "Thinking degrades the result in both; the relative degradation out of distribution is "
        "larger.",
    ),
    (
        s(-0.1, 0.2),
        s(-0.1, 0.25),
        s(0.0, 0.1),
        "degrades_both_same",
        "Thinking degrades the result in distribution on average and out of distribution on "
        "average; the relative degradation out of distribution is not measurably different.",
    ),
    (s(0.3, 0.1), s(-0.2, 0.1), s(-0.5, 0.1), "degrades_ood_only", DEGRADES_OOD_ONLY),
    (s(0.1, 0.05), s(0.3, 0.4), s(0.2, 0.1), "difference_only", DIFFERENCE_ONLY),
]


@pytest.mark.parametrize(("i_id", "i_ood", "delta", "key", "expected"), OUTCOMES_3_TO_5)
def test_outcomes_3_to_5_say_they_are_not_the_papers_effect_for_text(
    i_id, i_ood, delta, key, expected
):
    assert report.outcome(i_id, i_ood, delta) == key
    assert report.outcome_text(key, "text", i_id, i_ood) == expected + NOT_PAPER


@pytest.mark.parametrize("task", ["mnist", "regression"])
@pytest.mark.parametrize(("i_id", "i_ood", "delta", "key", "expected"), OUTCOMES_3_TO_5)
def test_outcomes_3_to_5_make_no_reference_to_the_paper_outside_text(
    task, i_id, i_ood, delta, key, expected
):
    assert report.outcome(i_id, i_ood, delta) == key
    text = report.outcome_text(key, task, i_id, i_ood)
    assert text == expected
    assert "paper" not in text


def test_the_adapter_returns_the_named_split_without_changing_the_data():
    data = regression_g2.make_data(FAST_REGRESSION, regression_g2.DATA_SEED)
    ood, test = g5.split_of(data, "ood"), g5.split_of(data, "test")
    assert ood.name == "ood" and ood.rows is data.ood
    assert test.name == "test" and test.rows is data.test
    assert g5.split_of(data, "ood").first(8).rows.context.equal(data.ood.context[:8])
    assert g5.split_of(data, "ood").first(None) == ood
    with pytest.raises(ValueError, match="unknown split"):
        g5.split_of(data, "validation")
    text = make_text_data(TEXT_CORPUS, None, text_lm.TIERS[Tier.FAST].sequence_length)
    with pytest.raises(ValueError, match="no ood split"):
        g5.split_of(text, "ood")


def regression_checkpoint(directory, model="ebt_mse", seed=0):
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    result, _, trained = regression_g2.run(candidate, Tier.FAST, seed, CPU, with_test=True)
    path = checkpoint_path(directory, model, seed)
    save_checkpoint(path, trained, code_fingerprint("regression_g2"), result)
    return path


def test_the_regression_curve_is_measured_on_the_out_of_distribution_split(tmp_path):
    path = regression_checkpoint(tmp_path)
    measured = g5.regression_curve(path, CPU, "ood", None)
    data = regression_g2.make_data(FAST_REGRESSION, regression_g2.DATA_SEED)
    assert measured["split"] == {"name": "ood", "rows": len(data.ood)}
    assert measured["x_range"][0] > 4 and measured["x_range"][1] <= 9
    # The G3 function of inference_regression on the out-of-distribution rows, computed here.
    setup = regression_g2.restore(path, CPU)
    expected = inference_regression.regression_g3(setup, data.ood, data.init_std, 0, CPU)
    assert measured["g3"] == expected
    # The same function on the test split gives the G3 of the existing evaluation, and differs.
    on_test = g5.regression_curve(path, CPU, "test", None)["g3"]
    existing = inference_regression.evaluate(path, CPU, None, None)["g3"]
    assert on_test == existing
    assert on_test["branch_distance"] != measured["g3"]["branch_distance"]


TEXT_CORPUS = "".join(f"Line {i % 7}: to be, or not to be; that is {i % 3}.\n" for i in range(300))
TEXT_OOD = (
    "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
    "Chapter I.]\n" + "It is a truth universally acknowledged, said she. " * 400 + "\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
)


@pytest.fixture(scope="module")
def text_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("text")
    data = make_text_data(TEXT_CORPUS, TEXT_OOD, text_lm.TIERS[Tier.FAST].sequence_length)
    candidate = text_lm.search_space("ebt", Tier.FAST, directory)[0]
    result, _, trained = text_lm.run(candidate, Tier.FAST, 0, CPU, True, data)
    path = checkpoint_path(directory, "ebt", 0)
    save_checkpoint(path, trained, code_fingerprint("text_lm"), result)
    return path, data


def test_the_text_curve_on_the_test_split_equals_the_existing_evaluation(text_checkpoint):
    path, data = text_checkpoint
    existing = inference_text.evaluate(path, CPU, data, None)["g3"]
    on_test = g5.text_curve(path, CPU, data, "test", None)["g3"]
    assert on_test["bpc"] == existing["bpc"]
    assert on_test["undefined_from_step"] == existing["undefined_from_step"]
    assert on_test["positions"] == data.test.target.numel()


def test_the_text_curve_is_measured_on_the_out_of_distribution_block(text_checkpoint):
    path, data = text_checkpoint
    measured = g5.text_curve(path, CPU, data, "ood", None)
    assert measured["split"] == {"name": "ood", "rows": len(data.ood)}
    assert measured["g3"]["positions"] == data.ood.target.numel()
    model, loop = text_lm.restore(path, CPU)
    batch = text_lm.TIERS[Tier.FAST].eval_batch
    assert measured["g3"] == g5.text_g3(model, loop, data.ood, batch, 0, CPU)
    on_test = g5.text_g3(model, loop, data.test, batch, 0, CPU)
    assert measured["g3"]["bpc"] != on_test["bpc"]


def test_the_noise_level_and_seed_are_applied():
    clean = torch.linspace(-1, 1, 2 * 28 * 28).reshape(2, 1, 28, 28)
    for level, seed in g5.MNIST_LEVELS.items():
        alpha = alpha_bar(level)
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
        expected = math.sqrt(alpha) * clean + math.sqrt(1 - alpha) * noise
        assert torch.allclose(g5.noisy_images(clean, level, seed), expected, rtol=0, atol=1e-6)
    # The training level and test seed reproduce the noisy test images of the final runs.
    reproduced = g5.noisy_images(
        clean, mnist_denoising.NOISE_LEVEL, mnist_denoising.TEST_NOISE_SEED
    )
    assert torch.equal(
        reproduced, mnist_denoising.fixed_noise(clean, mnist_denoising.TEST_NOISE_SEED)
    )


def test_the_noise_scale_grows_with_the_level():
    clean = torch.zeros(64, 1, 28, 28)
    for level, scale in ((0.15, 0.4600), (0.2, 0.5833), (0.3, 0.7773)):
        noisy = g5.noisy_images(clean, level, g5.MNIST_LEVELS[level])
        assert noisy.std().item() == pytest.approx(scale, rel=0.01)
    assert g5.MNIST_LEVELS == {0.15: 40_002, 0.2: 40_003, 0.3: 40_004}
    assert g5.PRIMARY_LEVEL == 0.2


@pytest.fixture(scope="module")
def mnist_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mnist")
    data = synthetic_images()
    candidate = mnist_denoising.search_space("ebt", Tier.FAST)[0]
    result, _, trained = mnist_denoising.run(candidate, Tier.FAST, 0, CPU, True, data)
    path = checkpoint_path(directory, "ebt", 0)
    save_checkpoint(path, trained, code_fingerprint("mnist_denoising"), result)
    return path, data


def test_mnist_curve_at_training_level_equals_existing_evaluation(mnist_checkpoint):
    """The mnist curve at the training level equals the existing evaluation."""
    path, data = mnist_checkpoint
    existing = inference_mnist.evaluate(path, CPU, data, None)["g3"]
    at_training = g5.mnist_curve(path, CPU, data, 0.1, mnist_denoising.TEST_NOISE_SEED, None)
    assert at_training["g3"]["psnr"] == existing["psnr"]


def test_the_mnist_curve_uses_the_given_noise_level(mnist_checkpoint):
    path, data = mnist_checkpoint
    measured = g5.mnist_curve(path, CPU, data, 0.2, 40_003, None)
    assert measured["split"]["noise_level"] == 0.2 and measured["split"]["noise_seed"] == 40_003
    model, loop = mnist_denoising.restore(path, CPU)
    noisy = g5.noisy_images(data.test.target, 0.2, 40_003)
    batch = mnist_denoising.TIERS[Tier.FAST].eval_batch
    assert measured["g3"] == g5.mnist_g3(model, loop, data.test.target, noisy, batch, CPU)
    at_training = g5.mnist_curve(path, CPU, data, 0.1, mnist_denoising.TEST_NOISE_SEED, None)
    assert measured["g3"]["psnr"][0] < at_training["g3"]["psnr"][0]  # more noise, lower PSNR


def test_the_g5_fingerprint_is_its_own_and_covers_the_task_modules():
    names = {path.name for path in fingerprint_files("inference_g5")}
    assert {
        "inference_g5.py",
        "inference_regression.py",
        "inference_mnist.py",
        "inference_text.py",
        "inference_g3_g4.py",
    } <= names
    for module in ("inference_regression", "inference_mnist", "inference_text"):
        assert "inference_g5.py" not in {p.name for p in fingerprint_files(module)}
    assert code_fingerprint("inference_g5") != code_fingerprint("inference_regression")


def test_results_are_written_skipped_and_stale_on_another_split(tmp_path):
    checkpoints, output = tmp_path / "checkpoints", tmp_path / "out"
    regression_checkpoint(checkpoints)
    summaries = g5.run_task(
        "regression",
        Tier.FAST,
        CPU,
        ["ebt_mse"],
        [0],
        checkpoints=checkpoints,
        output=output,
        rows=32,
    )
    stored = read_json(output / "ebt_mse-seed0.json")
    assert stored["experiment"] == "inference_g5" and stored["question"] == "G5"
    assert stored["code_fingerprint"] == code_fingerprint("inference_g5")
    assert stored["split"] == {"name": "ood", "rows": 32}
    assert stored["settings"] == {"name": "ood"}
    assert "g4" not in stored
    assert summaries[str(output)]["models"]["ebt_mse"]["g3"]["metric"] == "branch_distance"
    before = (output / "ebt_mse-seed0.json").read_bytes()
    g5.run_task(
        "regression",
        Tier.FAST,
        CPU,
        ["ebt_mse"],
        [0],
        checkpoints=checkpoints,
        output=output,
        rows=32,
    )
    assert (output / "ebt_mse-seed0.json").read_bytes() == before  # skipped
    with pytest.raises(runner.StaleResultsError):
        g5.run_task(
            "regression",
            Tier.FAST,
            CPU,
            ["ebt_mse"],
            [0],
            checkpoints=checkpoints,
            output=output,
            rows=16,
        )
    write_json(output / "ebt_mse-seed0.json", {**stored, "settings": {"name": "test"}})
    with pytest.raises(runner.StaleResultsError):
        g5.run_task(
            "regression",
            Tier.FAST,
            CPU,
            ["ebt_mse"],
            [0],
            checkpoints=checkpoints,
            output=output,
            rows=32,
        )


G5_FINGERPRINT = code_fingerprint("inference_g5")


def fake(seed, values, metric="branch_distance", sha="a", settings=None, **extra):
    return {
        "seed": seed,
        "code_fingerprint": G5_FINGERPRINT,
        "test_rows_limit": None,
        "settings": settings or {"name": "ood"},
        "checkpoint": {"path": f"seed{seed}.pt", "sha256": sha * 64},
        "trained_inference": {"steps": 4},
        "g3": {"steps": list(core.STEP_GRID), metric: values},
        **extra,
    }


def curve(first, trained, last):
    return [first, first, trained, trained, last, last, last]


REGRESSION_MODELS = g5.TASKS["regression"].models


def write_regression(results, inside_last=(0.4,) * 5, outside_last=(0.4, 0.3, 0.2, 0.3, 0.3)):
    """I_ID = 1 - last / 0.8 and I_OOD = 1 - last / 1.0 for every regression EBT."""
    inside = results / "regression" / "g3_g4" / "full"
    outside = results / "regression" / "g5" / "full"
    for model in REGRESSION_MODELS:
        for seed in range(5):
            write_json(
                inside / f"{model}-seed{seed}.json", fake(seed, curve(0.8, 0.6, inside_last[seed]))
            )
            write_json(
                outside / f"{model}-seed{seed}.json",
                fake(seed, curve(1.0, 0.5, outside_last[seed])),
            )
    return inside, outside


def test_the_report_computes_i_delta_and_the_outcome_and_writes_the_table(tmp_path):
    results = tmp_path / "results"
    write_regression(results)
    produced = report.task_report(results, "regression", "full")
    assert [e["model"] for e in produced["comparisons"]] == list(REGRESSION_MODELS)
    entry = produced["comparisons"][0]
    assert entry["i_id"]["values"] == pytest.approx([0.5] * 5)
    assert entry["i_ood"]["values"] == pytest.approx([0.6, 0.7, 0.8, 0.7, 0.7])
    assert entry["delta"]["mean"] == pytest.approx(0.2)
    assert entry["outcome"] == "helps_more"
    assert entry["outcome_text"] == report.OUTCOMES["helps_more"]
    assert entry["branch_distance_ood_4"]["mean"] == pytest.approx(0.5)
    text = report.table(produced)
    assert "| ebt_mse | split | 0.5000 ± 0.0000 |" in text
    assert report.OUTCOMES["helps_more"] in text


def test_delta_in_the_report_is_taken_per_seed_not_from_the_mean_of_i_id(tmp_path):
    # I_ID varies by seed and I_OOD = I_ID + 0.1 on every seed: Δ has a zero-width interval.
    results = tmp_path / "results"
    write_regression(
        results, inside_last=(0.8, 0.6, 0.4, 0.2, 0.4), outside_last=(0.9, 0.65, 0.4, 0.15, 0.4)
    )
    entry = report.task_report(results, "regression", "full")["comparisons"][0]
    assert entry["delta"]["values"] == pytest.approx([0.1] * 5)
    assert entry["delta"]["half_width"] == pytest.approx(0.0, abs=1e-12)


def test_a_degradation_in_both_is_not_reported_as_helping_more(tmp_path):
    # e rises in both (I < 0), less out of distribution: Δ > 0, but not helping more.
    results = tmp_path / "results"
    write_regression(
        results, inside_last=(1.6, 1.7, 1.6, 1.5, 1.6), outside_last=(1.2, 1.3, 1.25, 1.2, 1.2)
    )
    entry = report.task_report(results, "regression", "full")["comparisons"][0]
    assert entry["i_id"]["mean"] < 0 and entry["i_ood"]["mean"] < 0
    assert report.above(entry["delta"])
    assert entry["outcome"] == "degrades_both_smaller"
    text = report.table(report.task_report(results, "regression", "full"))
    assert "Thinking degrades the result in both" in text
    assert "paper" not in entry["outcome_text"]


def test_a_missing_out_of_distribution_file_is_an_error_naming_it(tmp_path):
    results = tmp_path / "results"
    _, outside = write_regression(results)
    (outside / "ebt_combined-seed3.json").unlink()
    with pytest.raises(report.G5ReportError, match="regression ebt_combined seed 3: missing"):
        report.task_report(results, "regression", "full")


def test_a_missing_in_distribution_file_is_an_error(tmp_path):
    results = tmp_path / "results"
    inside, _ = write_regression(results)
    (inside / "ebt_mse-seed0.json").unlink()
    with pytest.raises(report.G5ReportError, match="ebt_mse seed 0: missing"):
        report.task_report(results, "regression", "full")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"code_fingerprint": "other code"}, "not from the current inference_g5 code"),
        ({"test_rows_limit": 64}, "limited to 64 rows"),
        ({"settings": {"name": "test"}}, "has settings"),
        ({"checkpoint": {"path": "x.pt", "sha256": "b" * 64}}, "different checkpoints"),
    ],
)
def test_a_stale_limited_or_mismatched_g5_file_is_an_error_naming_it(tmp_path, change, message):
    results = tmp_path / "results"
    _, outside = write_regression(results)
    path = outside / "ebt_regularized-seed1.json"
    write_json(path, {**read_json(path), **change})
    with pytest.raises(report.G5ReportError, match=message) as raised:
        report.task_report(results, "regression", "full")
    assert "ebt_regularized-seed1.json" in str(raised.value)


MNIST_LAST = {0.15: 18.0, 0.2: 23.0, 0.3: 21.0}
"""PSNR after 64 steps out of distribution per level: only 0.2 improves on the 22.0 of step 1,
so each level gives another outcome and the primary one can be told from the others."""


def write_mnist(results, levels=tuple(g5.MNIST_LEVELS)):
    inside = results / "mnist" / "g3_g4" / "full"
    for seed in range(5):
        write_json(inside / f"ebt-seed{seed}.json", fake(seed, curve(27.0, 27.5, 20.0), "psnr"))
        for level in levels:
            settings = report.expected_settings("mnist", level)
            directory = g5.level_directory(results / "mnist" / "g5" / "full", level)
            values = curve(22.0, 22.5, MNIST_LAST[level] + seed * 0.1)
            write_json(
                directory / f"ebt-seed{seed}.json", fake(seed, values, "psnr", settings=settings)
            )


def test_only_primary_mnist_level_has_outcome_others_labelled(tmp_path):
    """Only the primary mnist level has an outcome and the others are labelled."""
    results = tmp_path / "results"
    write_mnist(results)
    produced = report.task_report(results, "mnist", "full")
    outcomes = {e["noise_level"]: e["outcome"] for e in produced["comparisons"]}
    assert outcomes[0.15] is None and outcomes[0.3] is None
    assert outcomes[0.2] in report.OUTCOMES or outcomes[0.2] in report.DEGRADATION
    first = next(e for e in produced["comparisons"] if e["noise_level"] == 0.2)
    assert first["i_ood"]["values"][0] == pytest.approx(1 - 10 ** ((22.0 - 23.0) / 10))
    text = report.table(produced)
    assert "| ebt | noise 0.15 (no decision) |" in text
    assert "| ebt | noise 0.3 (no decision) |" in text
    assert "| ebt | noise 0.2 |" in text and "noise 0.2 (no decision)" not in text


def test_an_mnist_file_of_another_level_is_an_error(tmp_path):
    results = tmp_path / "results"
    write_mnist(results)
    path = g5.level_directory(results / "mnist" / "g5" / "full", 0.2) / "ebt-seed4.json"
    write_json(path, {**read_json(path), "settings": report.expected_settings("mnist", 0.15)})
    with pytest.raises(report.G5ReportError, match=r"mnist ebt level 0\.2 seed 4: .*has settings"):
        report.task_report(results, "mnist", "full")


def test_a_missing_mnist_level_is_an_error_naming_the_level(tmp_path):
    results = tmp_path / "results"
    write_mnist(results, levels=(0.15, 0.2))
    with pytest.raises(report.G5ReportError, match=r"mnist ebt level 0\.3 seed 0: missing"):
        report.task_report(results, "mnist", "full")


def test_the_expected_settings_are_those_inference_g5_records():
    assert report.expected_settings("regression", None) == {"name": "ood"}
    assert report.expected_settings("text", None) == {"name": "ood"}
    assert report.expected_settings("mnist", 0.2) == {
        "name": "test",
        "noise_level": 0.2,
        "noise_seed": 40_003,
    }


def test_the_mnist_outcome_is_computed_from_the_noise_0_2_data(tmp_path):
    results = tmp_path / "results"
    write_mnist(results)
    produced = report.task_report(results, "mnist", "full")
    by_level = {e["noise_level"]: e for e in produced["comparisons"]}
    primary = [e for e in produced["comparisons"] if e["primary"]]
    assert [e["noise_level"] for e in primary] == [0.2]
    for level, entry in by_level.items():
        expected = [1 - 10 ** ((22.0 - MNIST_LAST[level] - seed * 0.1) / 10) for seed in range(5)]
        assert entry["i_ood"]["values"] == pytest.approx(expected)
    # Only level 0.2 improves out of distribution; 0.15 and 0.3 would degrade in both.
    assert report.outcome(*(by_level[0.15][k] for k in ("i_id", "i_ood", "delta"))) != "helps_more"
    assert by_level[0.2]["outcome"] == "helps_more"
    assert by_level[0.2]["outcome_text"] == (
        "Thinking helps out of distribution and degrades the result in distribution; Δ "
        "therefore includes the in-distribution degradation."
    )
    # The out-of-distribution rows of the error table are the out-of-distribution values.
    assert by_level[0.2]["psnr_ood_1"]["mean"] == pytest.approx(22.0)
    assert by_level[0.2]["psnr_id_1"]["mean"] == pytest.approx(27.0)
    assert by_level[0.2]["psnr_ood_64"]["mean"] == pytest.approx(23.2)


def test_an_mnist_file_with_another_noise_seed_is_an_error(tmp_path):
    results = tmp_path / "results"
    write_mnist(results)
    path = g5.level_directory(results / "mnist" / "g5" / "full", 0.2) / "ebt-seed2.json"
    settings = {**report.expected_settings("mnist", 0.2), "noise_seed": 40_002}
    write_json(path, {**read_json(path), "settings": settings})
    with pytest.raises(report.G5ReportError, match=r"mnist ebt level 0\.2 seed 2: .*has settings"):
        report.task_report(results, "mnist", "full")


def write_text(results):
    """Text: BPC falls by 0.1 in distribution and by about 0.3 out of distribution."""
    inside = results / "text" / "g3_g4" / "full"
    outside = results / "text" / "g5" / "full"
    for model in g5.TASKS["text"].models:
        for seed in range(5):
            write_json(
                inside / f"{model}-seed{seed}.json",
                fake(seed, curve(2.4, 2.35, 2.3 + seed * 0.001), "bpc"),
            )
            write_json(
                outside / f"{model}-seed{seed}.json",
                fake(seed, curve(2.8, 2.6, 2.5 + seed * 0.002), "bpc"),
            )


def test_the_report_names_the_papers_effect_for_text_and_not_for_regression(tmp_path):
    results = tmp_path / "results"
    write_text(results)
    write_regression(results)
    text_entry = report.task_report(results, "text", "full")["comparisons"][0]
    assert text_entry["outcome"] == "helps_more"
    assert text_entry["outcome_text"] == HELPS_MORE + PAPER
    regression = report.task_report(results, "regression", "full")
    assert regression["comparisons"][0]["outcome_text"] == HELPS_MORE
    assert "paper reports" not in report.table(regression)


def drop_step_64(data):
    g3 = dict(data["g3"])
    metric = next(k for k in g3 if k != "steps")
    g3["steps"], g3[metric] = g3["steps"][:-1], g3[metric][:-1]
    return {**data, "g3": g3}


@pytest.mark.parametrize("side", ["inside", "outside"])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: {k: v for k, v in d.items() if k != "checkpoint"}, "has no field checkpoint"),
        (lambda d: {**d, "checkpoint": {"path": "x.pt"}}, "has no field checkpoint.sha256"),
        (lambda d: {k: v for k, v in d.items() if k != "g3"}, "has no field g3"),
        (drop_step_64, "has no field g3 step 64"),
        (lambda d: {**d, "g3": {"steps": d["g3"]["steps"]}}, "has no field g3.branch_distance"),
    ],
)
def test_a_missing_field_is_an_error_naming_the_file_and_the_field(tmp_path, side, change, message):
    results = tmp_path / "results"
    folders = dict(zip(("inside", "outside"), write_regression(results), strict=True))
    path = folders[side] / "ebt_combined-seed2.json"
    write_json(path, change(read_json(path)))
    with pytest.raises(report.G5ReportError, match=message) as raised:
        report.task_report(results, "regression", "full")
    assert str(path) in str(raised.value)
