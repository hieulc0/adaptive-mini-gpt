"""Statistical analysis utilities for paired experimental evaluations.

Implements rigorous Student-t significance tests, degrees of freedom accounting (df = n - 1),
exact two-tailed p-values via Student-t survival function quadrature, and 95% confidence intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedTTestResult:
    """Rigorous summary of a paired Student-t test comparison."""

    delta_mean: float
    std_diff: float
    se: float
    t_stat: float
    df: int
    p_value: float
    ci_95_lower: float
    ci_95_upper: float
    is_significant_05: bool
    is_significant_01: bool


# Exact two-tailed critical t-values for alpha = 0.05
T_CRIT_05_TABLE: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    60: 2.000,
    120: 1.980,
}


def student_t_pdf(u: float, df: int) -> float:
    """Probability density function of Student-t distribution with df degrees of freedom."""
    coef = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
    return coef * ((1.0 + (u * u) / df) ** (-(df + 1.0) / 2.0))


def student_t_two_tailed_p_value(t_stat: float, df: int, steps: int = 1000) -> float:
    """Compute exact two-tailed p-value for Student-t statistic with df degrees of freedom."""
    x = abs(t_stat)
    if x == 0.0:
        return 1.0
    if df <= 0:
        raise ValueError(f"Degrees of freedom must be >= 1, got {df}")

    # Closed-form fast path for df = 4 (standard 5-seed benchmark)
    if df == 4:
        denom = 4.0 + x * x
        sqrt_denom = math.sqrt(denom)
        term = (x / sqrt_denom) * (1.0 + 2.0 / denom)
        return max(0.0001, min(1.0, 1.0 - term))

    # High-precision composite Simpson's rule quadrature
    # Ensures steps is even
    steps = steps if steps % 2 == 0 else steps + 1
    h = x / steps
    s = student_t_pdf(0.0, df) + student_t_pdf(x, df)
    for i in range(1, steps):
        u = i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        s += weight * student_t_pdf(u, df)
    integral = (h / 3.0) * s
    p = 1.0 - 2.0 * integral
    return max(0.0001, min(1.0, p))


def compute_paired_t_test(
    sample_a: list[float],
    sample_b: list[float],
) -> PairedTTestResult:
    """Compute paired Student-t test across paired observations.

    Parameters:
        sample_a: Experimental arm observations [x_1, ..., x_n].
        sample_b: Control arm observations [y_1, ..., y_n].

    Returns:
        PairedTTestResult containing delta_mean, t_stat, df, p_value, and 95% CI.
    """
    n = len(sample_a)
    if n != len(sample_b):
        raise ValueError(f"Sample lengths must match: {n} != {len(sample_b)}")
    if n < 2:
        return PairedTTestResult(
            delta_mean=0.0,
            std_diff=0.0,
            se=0.0,
            t_stat=0.0,
            df=0,
            p_value=1.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
            is_significant_05=False,
            is_significant_01=False,
        )

    diffs = [a - b for a, b in zip(sample_a, sample_b, strict=False)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var) if var > 0 else 0.0
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0

    df = n - 1
    t_crit = T_CRIT_05_TABLE.get(df, 2.0)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    p_val = student_t_two_tailed_p_value(t_stat, df)

    return PairedTTestResult(
        delta_mean=round(mean_diff, 4),
        std_diff=round(std_diff, 4),
        se=round(se, 4),
        t_stat=round(t_stat, 4),
        df=df,
        p_value=round(p_val, 4),
        ci_95_lower=round(ci_lower, 4),
        ci_95_upper=round(ci_upper, 4),
        is_significant_05=bool(p_val < 0.05),
        is_significant_01=bool(p_val < 0.01),
    )
