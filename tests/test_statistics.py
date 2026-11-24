"""Student t quantiles and intervals (questions.md: intervals over seeds)."""

import math
from statistics import NormalDist

import pytest

from ebt.statistics import mean_interval, student_t_quantile


def test_one_degree_of_freedom_has_the_cauchy_quantile():
    """With ν = 1 the t distribution is Cauchy, whose 0.975 quantile is tan(0.475π) = 12.7062..."""
    assert student_t_quantile(0.975, 1) == pytest.approx(math.tan(0.475 * math.pi), rel=1e-6)


def test_two_degrees_of_freedom_have_the_closed_form_quantile():
    """With ν = 2 the quantile is (2p - 1)·sqrt(2 / (1 - (2p - 1)**2)), 4.3027 at p = 0.975."""
    p = 0.975
    a = 2 * p - 1
    assert student_t_quantile(p, 2) == pytest.approx(a * math.sqrt(2 / (1 - a * a)), rel=1e-6)


def test_four_degrees_of_freedom_have_the_tabulated_quantile():
    """ν = 4 gives every interval over 5 seeds; t(0.975, 4) = 2.776445 to 6 decimal places."""
    assert round(student_t_quantile(0.975, 4), 6) == 2.776445


def test_large_degrees_of_freedom_approach_the_normal_quantile():
    """At ν = 10 000 the t quantile is within 1e-3 of the normal quantile 1.959964."""
    normal = NormalDist().inv_cdf(0.975)
    assert student_t_quantile(0.975, 10_000) == pytest.approx(normal, abs=1e-3)


def test_interval_of_five_values():
    mean, half = mean_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert half == pytest.approx(student_t_quantile(0.975, 4) * math.sqrt(2.5 / 5))
