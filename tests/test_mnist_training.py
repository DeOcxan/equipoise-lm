"""MNIST denoising experiment on the fast tier with synthetic images (method.md section 17)."""

import pytest
import torch

from ebt.config import Tier
from ebt.experiments import mnist_denoising, runner
from ebt.results import read_json
from support import synthetic_images

CPU = torch.device("cpu")


DATA = synthetic_images()


@pytest.mark.parametrize("model", mnist_denoising.MODELS)
def test_fast_tier_training_loss_falls_and_the_validation_curve_is_recorded(model):
    candidate = mnist_denoising.search_space(model, Tier.FAST)[0]
    result, (name, value), _ = mnist_denoising.run(candidate, Tier.FAST, 0, CPU, True, DATA)
    losses = result.history["train_loss"]
    assert losses[-1] < losses[0]
    settings = mnist_denoising.TIERS[Tier.FAST]
    steps = list(range(settings.validate_every, settings.train_steps + 1, settings.validate_every))
    assert result.history["validation_steps"] == steps
    assert len(result.history["validation_psnr"]) == len(steps)
    assert result.history["validation_smooth_l1"][-1] == pytest.approx(value)
    assert name == "validation_smooth_l1"
    assert set(result.metrics) == {"validation", "test"}
    assert len(result.artifacts["examples"]["outputs"]) == (2 if model == "ebt" else 1)


def test_pass_counts_follow_method_section_17_6():
    expected = {"ebt": ((1, 1, 1), (2, 2, 0)), "denoiser": ((1, 1, 0), (1, 0, 0))}
    for model, (train, inference) in expected.items():
        candidate = mnist_denoising.search_space(model, Tier.FAST)[0]
        result, _, _ = mnist_denoising.run(candidate, Tier.FAST, 1, CPU, False, DATA)
        examples = result.cost["train_examples"]
        total = result.cost["train_total_example_passes"]
        per_example = tuple(
            total[key] / examples for key in ("forward", "backward", "second_order")
        )
        assert per_example == train
        passes = result.cost["inference_example_passes_per_example"]
        assert (passes["forward"], passes["backward"], passes["second_order"]) == inference


def test_noisy_input_psnr_is_the_same_for_every_run():
    """Validation and test noise are fixed, so the noisy-input reference does not vary."""
    values = set()
    for seed in (0, 1):
        candidate = mnist_denoising.search_space("denoiser", Tier.FAST)[0]
        result, _, _ = mnist_denoising.run(candidate, Tier.FAST, seed, CPU, False, DATA)
        values.add(result.metrics["validation"]["noisy_input_psnr"])
    assert len(values) == 1


def test_rerun_skips_work_on_disk(tmp_path):
    def train_one(candidate, seed, with_test):
        return mnist_denoising.run(candidate, Tier.FAST, seed, CPU, with_test, DATA)

    def space(model):
        return mnist_denoising.search_space(model, Tier.FAST)

    models, seeds = ("denoiser",), (0, 1)

    def execute(run):
        runner.execute(
            tmp_path,
            "all",
            models,
            space,
            mnist_denoising.Candidate,
            run,
            mnist_denoising.SEARCH_SEED,
            seeds,
            runner.Provenance("current code"),
            tmp_path / "checkpoints",
        )
        return read_json(tmp_path / "search.json")

    record = execute(train_one)

    def fail(*args):
        raise AssertionError("work on disk was repeated")

    again = execute(fail)
    assert again["models"] == record["models"]
