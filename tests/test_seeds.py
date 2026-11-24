"""Every random source of every run of an experiment has its own seed (method.md 16.6)."""

import pytest

from ebt.experiments import mnist_denoising, regression_g2, text_lm
from ebt.runtime import run_seed


@pytest.mark.parametrize("experiment", [regression_g2, mnist_denoising, text_lm])
def test_no_two_random_sources_share_a_seed(experiment):
    """Over the data seeds, the fixed evaluation seeds, the search seed and the reported seeds,
    every seed of every random source occurs once."""
    seeds = list(experiment.FIXED_SEEDS.values())
    for run in (experiment.SEARCH_SEED, *experiment.REPORT_SEEDS):
        sources = experiment.run_seeds(run)
        assert tuple(sources) == experiment.RUN_SOURCES
        seeds += sources.values()
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize("experiment", [regression_g2, mnist_denoising, text_lm])
def test_initialization_has_its_own_source(experiment):
    """Parameter initialization (the global generator) is one of the sources."""
    assert "initialization" in experiment.RUN_SOURCES


def test_run_seed_rejects_seeds_outside_its_range():
    with pytest.raises(ValueError):
        run_seed(0, 1_000)
