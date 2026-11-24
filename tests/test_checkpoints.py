"""A saved and reloaded final-run model gives identical predictions on the CPU."""

import math

import pytest
import torch

from ebt.checkpoints import load_checkpoint, save_checkpoint
from ebt.config import Tier
from ebt.experiments import mnist_denoising, regression_g2
from ebt.runtime import make_generator
from ebt.tasks.mnist import denoising_metrics
from ebt.tasks.regression import make_data, regression_metrics
from ebt.training.batch import Batch
from ebt.training.evaluation import predict_ebt, predict_feedforward
from support import synthetic_images

CPU = torch.device("cpu")
SEED = 2
DATA = synthetic_images()


@pytest.mark.parametrize("model", regression_g2.MODELS)
def test_regression_model_reloads_with_identical_predictions(tmp_path, model):
    candidate = regression_g2.search_space(model, Tier.FAST)[0]
    result, _, trained = regression_g2.run(candidate, Tier.FAST, SEED, CPU, with_test=False)
    path = tmp_path / f"{model}-seed{SEED}.pt"
    save_checkpoint(path, trained, "current code", result)

    restored = regression_g2.restore(path, CPU)
    data = make_data(regression_g2.TIERS[Tier.FAST].data, regression_g2.DATA_SEED)
    evaluation = make_generator(regression_g2.run_seeds(SEED)["evaluation"], CPU)
    predictions = restored.predict(data.validation, evaluation)
    metrics = regression_metrics(data.validation, predictions.final, predictions.initial)

    saved = load_checkpoint(path)
    assert saved["code_fingerprint"] == "current code"
    assert saved["run"]["seed"] == SEED
    for key, value in metrics.items():
        stored = result.metrics["validation"][key]
        assert value == stored or (math.isnan(value) and math.isnan(stored)), key


@pytest.mark.parametrize("model", mnist_denoising.MODELS)
def test_denoising_model_reloads_with_identical_predictions(tmp_path, model):
    candidate = mnist_denoising.search_space(model, Tier.FAST)[0]
    result, _, trained = mnist_denoising.run(candidate, Tier.FAST, SEED, CPU, False, DATA)
    path = tmp_path / f"{model}-seed{SEED}.pt"
    save_checkpoint(path, trained, "current code", result)

    network, loop = mnist_denoising.restore(path, CPU)
    clean = DATA.validation.target
    noisy = mnist_denoising.fixed_noise(clean, mnist_denoising.VALIDATION_NOISE_SEED)
    rows = mnist_denoising.TIERS[Tier.FAST].eval_batch
    if loop is None:
        final = predict_feedforward(network, noisy, rows).final
    else:
        batch = Batch(None, noisy)
        inference = mnist_denoising.INFERENCE
        final = predict_ebt(network, loop, batch, noisy, inference, None, rows, False).final
    metrics = denoising_metrics(final, clean)

    assert metrics["smooth_l1"] == result.metrics["validation"]["smooth_l1"]
    assert metrics["psnr"] == result.metrics["validation"]["psnr"]
