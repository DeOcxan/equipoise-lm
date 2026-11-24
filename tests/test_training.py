"""Shared training infrastructure on the fast tier (method.md sections 10 and 16)."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from ebt.config import (
    MLPConfig,
    ObjectiveConfig,
    ReplayConfig,
    ThinkingConfig,
    Tier,
    TrainingConfig,
    TrainingMode,
)
from ebt.core.cost import PassCount
from ebt.core.replay import ReplayBuffer
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import regression_g2, runner
from ebt.models.mlp import MLPEnergy
from ebt.results import read_json, write_result
from ebt.training.batch import Batch
from ebt.training.objectives import EBTObjective
from ebt.training.optimizer import build_optimizer

CPU = torch.device("cpu")


@pytest.mark.parametrize("model", regression_g2.MODELS)
def test_fast_tier_training_loss_falls(model):
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    result, _, _ = regression_g2.run(candidate, Tier.FAST, seed=0, device=CPU, with_test=False)
    losses = result.history["train_loss"]
    assert losses[-1] < losses[0]


def test_step_size_group_has_multiplied_lr_and_no_weight_decay():
    """Step size group has the multiplied learning rate and no weight decay."""
    model = MLPEnergy(1, 1, MLPConfig(widths=(8,)))
    loop = ThinkingLoop(ThinkingConfig(step_size=3000.0, step_size_learnable=True))
    config = TrainingConfig(
        learning_rate=1e-4,
        train_steps=10,
        batch_size=4,
        warmup_steps=0,
        weight_decay=0.01,
        step_size_lr_multiplier=9000.0,
    )
    optimizer = build_optimizer(torch.nn.ModuleList([model, loop]), loop, config)
    model_group, alpha_group = optimizer.param_groups
    assert alpha_group["params"] == [loop.step_size.raw]
    assert alpha_group["lr"] == pytest.approx(1e-4 * 9000.0)
    assert alpha_group["weight_decay"] == 0.0
    assert model_group["lr"] == pytest.approx(1e-4) and model_group["weight_decay"] == 0.01
    assert all(parameter is not loop.step_size.raw for parameter in model_group["params"])


def test_fixed_step_size_gets_no_group():
    model = MLPEnergy(1, 1, MLPConfig(widths=(8,)))
    loop = ThinkingLoop(ThinkingConfig(step_size=0.3))
    config = TrainingConfig(learning_rate=1e-3, train_steps=10, batch_size=4, warmup_steps=0)
    optimizer = build_optimizer(torch.nn.ModuleList([model, loop]), loop, config)
    assert len(optimizer.param_groups) == 1


@pytest.mark.parametrize("model", ["ebt_contrastive", "regressor"])
def test_same_seed_on_cpu_gives_an_identical_result_file(model, tmp_path):
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    paths = []
    for name in ("first", "second"):
        result, _, _ = regression_g2.run(candidate, Tier.FAST, seed=3, device=CPU, with_test=True)
        paths.append(tmp_path / f"{name}.json")
        write_result(paths[-1], result)
    assert paths[0].read_bytes() == paths[1].read_bytes()


@pytest.mark.parametrize(
    ("objective", "replay", "expected"),
    [
        # 3 steps in S2 per row: forward 3, backward 3, second order 1.
        (ObjectiveConfig(1.0, 0.0), False, PassCount(3, 3, 1) * 8),
        # The contrastive term adds one forward and one backward pass per row.
        (ObjectiveConfig(0.0, 1.0), False, PassCount(4, 4, 1) * 8),
        # The energy penalty reuses the real and fake energies: no further pass.
        (ObjectiveConfig(0.0, 1.0, 0.1), False, PassCount(4, 4, 1) * 8),
        # With a full buffer and fraction 0.5, 4 replayed rows join the 8 fresh rows.
        (ObjectiveConfig(1.0, 0.0), True, PassCount(3, 3, 1) * 12),
    ],
)
def test_ebt_objective_pass_counts(objective, replay, expected):
    model = MLPEnergy(1, 1, MLPConfig(widths=(8,)))
    loop = ThinkingLoop(
        ThinkingConfig(mode=TrainingMode.S2, step_size=0.1, steps_min=3, steps_max=3)
    )
    buffer = None
    if replay:
        buffer = ReplayBuffer(
            ReplayConfig(capacity=16, fraction=0.5), torch.Generator().manual_seed(0)
        )
    ebt = EBTObjective(
        model, loop, F.mse_loss, objective, lambda b, g: torch.zeros_like(b.target), buffer
    )
    batch = Batch(torch.rand(8, 1), torch.rand(8, 1))
    generator = torch.Generator().manual_seed(1)
    if replay:
        ebt.batch_loss(batch, generator)  # fills the buffer with 8 rows
    assert ebt.batch_loss(batch, generator).cost == expected


def test_result_records_the_configuration_and_costs():
    candidate = regression_g2.search_space("ebt_mse", Tier.FAST)[0]
    result, (name, value), _ = regression_g2.run(
        candidate, Tier.FAST, seed=0, device=CPU, with_test=True
    )
    assert name == "validation_mse" and value == result.metrics["validation"]["validation_mse"]
    assert result.config["thinking"]["mode"] == "s2"
    assert result.parameters["trainable"] == 4416
    assert set(result.artifacts) == {"predictions", "energy_grid", "energy_gradient"}
    steps = dataclasses.asdict(regression_g2.TIERS[Tier.FAST])["train_steps"]
    assert result.cost["train_examples"] == steps * regression_g2.TIERS[Tier.FAST].batch_size


def test_search_and_final_skip_work_already_on_disk(tmp_path):
    """A rerun reads finished configurations and runs from disk instead of training again."""
    models = ("regressor",)

    def space(model):
        return regression_g2.search_space(model, Tier.FAST)

    def train_one(candidate, seed, with_test):
        return regression_g2.run(candidate, Tier.FAST, seed, CPU, with_test)

    seeds, seed = regression_g2.REPORT_SEEDS, regression_g2.SEARCH_SEED
    provenance = runner.Provenance("current code")

    def execute(run):
        runner.execute(
            tmp_path,
            "all",
            models,
            space,
            regression_g2.Candidate,
            run,
            seed,
            seeds,
            provenance,
            tmp_path / "checkpoints",
        )
        return read_json(tmp_path / "search.json")

    record = execute(train_one)
    written = sorted(path.name for path in tmp_path.rglob("*.json"))
    assert len(written) == 1 + 1 + len(seeds)

    def fail(*args, **kwargs):
        raise AssertionError("work on disk was repeated")

    again = execute(fail)
    before, after = record["models"]["regressor"], again["models"]["regressor"]
    assert after["chosen"] == before["chosen"]
    assert [e["selection_value"] for e in after["tried"]] == [
        e["selection_value"] for e in before["tried"]
    ]
