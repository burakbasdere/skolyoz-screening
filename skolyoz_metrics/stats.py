"""Small dependency-free statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

# Two-sided 95 % normal quantile.
Z95 = 1.959963984540054


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile with linear interpolation (numpy's default definition)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Behaves sensibly at the extremes (e.g. 4/4), unlike the normal
    approximation, which collapses to a zero-width interval there.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Count, spread and tail percentiles of a sample (used for latencies)."""
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "min": min(values),
        "mean": mean,
        "std": math.sqrt(var),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }
