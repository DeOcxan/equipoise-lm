"""Mean and confidence interval over seeds (questions.md)."""

import math
from collections.abc import Sequence

SIMPSON_INTERVALS = 2000
"""Even number of Simpson intervals for integrating the Student t density."""


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """The `probability` quantile of Student's t distribution, for probability in (0.5, 1).

    Computed by bisection on the distribution function, which is 1/2 plus the integral of the
    density f(t) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + t²/ν)^(−(ν+1)/2) from 0, evaluated with
    Simpson's rule.
    """
    if not 0.5 < probability < 1 or degrees_of_freedom < 1:
        raise ValueError("probability must lie in (0.5, 1) and degrees_of_freedom be >= 1")
    nu = degrees_of_freedom
    log_norm = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)

    def density(t: float) -> float:
        return math.exp(log_norm - (nu + 1) / 2 * math.log1p(t * t / nu))

    def distribution(upper: float) -> float:
        h = upper / SIMPSON_INTERVALS
        total = density(0.0) + density(upper)
        for i in range(1, SIMPSON_INTERVALS):
            total += (4 if i % 2 else 2) * density(i * h)
        return 0.5 + total * h / 3

    low, high = 0.0, 1.0
    while distribution(high) < probability:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if distribution(middle) < probability else (low, middle)
    return (low + high) / 2


def mean_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the two-sided t confidence interval of the mean.

    The half-width is t_{(1+confidence)/2, n−1} · s / √n, with s the sample standard deviation.
    """
    n = len(values)
    if n < 2:
        raise ValueError("at least two values are needed for an interval")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    t = student_t_quantile((1 + confidence) / 2, n - 1)
    return mean, t * math.sqrt(variance / n)
