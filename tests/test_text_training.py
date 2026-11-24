"""Text language-model experiment on the fast tier with a synthetic corpus (method.md 18)."""

import pytest
import torch

from ebt.checkpoints import save_checkpoint
from ebt.config import Tier
from ebt.core.thinking import ThinkingLoop
from ebt.experiments import runner, text_lm
from ebt.models.text_transformer import CausalTransformer, TextEnergy
from ebt.results import read_json, read_result
from ebt.tasks.text import bits_per_character, make_data

CPU = torch.device("cpu")
LENGTH = text_lm.TIERS[Tier.FAST].sequence_length
CORPUS = "".join(
    f"Line {i % 7}: to be, or not to be; that is the question {i % 3}.\n" for i in range(300)
)
OOD = (
    "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
    "Chapter I.]\n" + "It is a truth universally acknowledged. " * 500 + "\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
)
DATA = make_data(CORPUS, OOD, LENGTH)


@pytest.mark.parametrize("model", text_lm.MODELS)
def test_fast_tier_trains_and_evaluates_the_best_checkpoint(tmp_path, model):
    candidate = text_lm.search_space(model, Tier.FAST, tmp_path)[0]
    result, (name, value), _ = text_lm.run(candidate, Tier.FAST, 0, CPU, True, DATA)
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
    assert name == "validation_bpc_best_checkpoint"
    assert value == min(result.history["validation_bpc"])
    assert result.history["best_step"] in result.history["validation_steps"]
    assert set(result.metrics) == {"validation", "test", "ood"}
    texts = result.artifacts["samples"]["texts"]
    assert len(texts) == text_lm.SAMPLES
    assert all(len(text) == text_lm.TIERS[Tier.FAST].sample_length for text in texts)


def test_ebt_passes_per_window_at_inference(tmp_path):
    candidate = text_lm.search_space("ebt", Tier.FAST, tmp_path)[0]
    result, _, _ = text_lm.run(candidate, Tier.FAST, 0, CPU, False, DATA)
    assert result.cost["inference_example_passes_per_example"] == {
        "forward": 4.0,
        "backward": 4.0,
        "second_order": 0.0,
    }


def test_second_ebt_stage_appears_once_the_first_is_on_disk(tmp_path):
    first = text_lm.search_space("ebt", Tier.FAST, tmp_path)
    settings = text_lm.TIERS[Tier.FAST]
    assert [c.step_size for c in first] == list(settings.step_sizes)
    values = [0.7, 0.3]
    for candidate, value in zip(first, values, strict=True):
        path = runner.search_path(tmp_path, "ebt", candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'{{"candidate": {{"model": "ebt", "learning_rate": {candidate.learning_rate}, '
            f'"step_size": {candidate.step_size}}}, "selection_value": {value}}}'
        )
    space = text_lm.search_space("ebt", Tier.FAST, tmp_path)
    assert space[:-1] == first
    assert space[-1] == text_lm.Candidate(
        "ebt", settings.second_stage_learning_rate, settings.step_sizes[1]
    )


def test_search_and_final_write_resumable_results_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(text_lm, "REPORT_SEEDS", (0,))
    output, checkpoints = tmp_path / "results", tmp_path / "checkpoints"
    arguments = (output, checkpoints, "all", ["transformer", "ebt"], Tier.FAST, CPU, DATA, False)
    text_lm.execute(*arguments)

    record = read_json(output / "search.json")
    tried = record["models"]["ebt"]["tried"]
    assert len(tried) == len(text_lm.TIERS[Tier.FAST].step_sizes) + 1
    for model in text_lm.MODELS:
        result = read_result(output / f"{model}-seed0.json")
        assert result.code["checkpoint"] == (checkpoints / f"{model}-seed0.pt").resolve().as_posix()
        assert (checkpoints / f"{model}-seed0.pt").exists()

    def fail(*args, **kwargs):
        raise AssertionError("work on disk was repeated")

    monkeypatch.setattr(text_lm, "run", fail)
    text_lm.execute(*arguments)


def test_final_stage_stops_when_the_second_ebt_stage_is_missing(tmp_path, monkeypatch):
    """An interrupted search: the first EBT stage and the baseline are on disk, the second EBT
    configuration is not."""
    monkeypatch.setattr(text_lm, "REPORT_SEEDS", (0,))
    output = tmp_path / "results"
    arguments = (tmp_path / "checkpoints", ["transformer", "ebt"], Tier.FAST, CPU, DATA, False)
    runner_execute = runner.execute
    calls = []

    def first_call_only(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            runner_execute(*args, **kwargs)

    monkeypatch.setattr(runner, "execute", first_call_only)
    text_lm.execute(output, arguments[0], "search", *arguments[1:])
    monkeypatch.setattr(runner, "execute", runner_execute)
    with pytest.raises(runner.IncompleteSearchError, match="incomplete"):
        text_lm.execute(output, arguments[0], "final", *arguments[1:])
    assert not list(output.glob("*-seed*.json"))


@pytest.mark.parametrize("model", text_lm.MODELS)
def test_saved_and_reloaded_model_gives_identical_predictions(tmp_path, model):
    candidate = text_lm.search_space(model, Tier.FAST, tmp_path)[0]
    result, _, trained = text_lm.run(candidate, Tier.FAST, 1, CPU, False, DATA)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained, "current code", result)

    network, loop = text_lm.restore(path, CPU)
    settings = text_lm.TIERS[Tier.FAST]
    seed = text_lm.FIXED_SEEDS["validation_initial"]
    predictions = text_lm.predict(network, loop, DATA.validation, seed, settings, CPU)
    bpc = bits_per_character(predictions.final, DATA.validation.target)
    assert bpc == result.metrics["validation"]["bpc"]


MPS_AVAILABLE = torch.backends.mps.is_available()


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS backend not available")
@pytest.mark.parametrize("model", text_lm.MODELS)
def test_generation_runs_on_mps(tmp_path, model):
    """MPS has no float64: generate() converts the logits on the CPU, where it samples."""
    mps = torch.device("mps")
    settings = text_lm.TIERS[Tier.FAST]
    config = text_lm.model_config(settings, len(DATA.vocabulary))
    candidate = text_lm.search_space(model, Tier.FAST, tmp_path)[0]
    torch.manual_seed(0)
    if model == "transformer":
        network, loop = CausalTransformer(config).to(mps), None
    else:
        network = TextEnergy(config).to(mps)
        loop = ThinkingLoop(text_lm.thinking_config(candidate)).to(mps)
    prompt = DATA.validation.context[0, : text_lm.PROMPT_LENGTH]
    generator = torch.Generator().manual_seed(0)
    samples = text_lm.generate(network, loop, prompt, settings, generator, mps)
    assert samples.device.type == "cpu"
    assert samples.shape == (text_lm.SAMPLES, settings.sample_length)
    assert samples.min() >= 0 and samples.max() < len(DATA.vocabulary)


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS backend not available")
@pytest.mark.parametrize("model", text_lm.MODELS)
def test_a_final_run_with_samples_runs_on_mps(tmp_path, model):
    candidate = text_lm.search_space(model, Tier.FAST, tmp_path)[0]
    seed = text_lm.REPORT_SEEDS[0]
    result, _, _ = text_lm.run(candidate, Tier.FAST, seed, torch.device("mps"), True, DATA)
    assert set(result.metrics) == {"validation", "test", "ood"}
    assert len(result.artifacts["samples"]["texts"]) == text_lm.SAMPLES
