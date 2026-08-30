"""Uncertainty for benchmark scores.

At n = 50 a two-point difference between two models is noise. Every number this pipeline
publishes carries an interval, and the leaderboard refuses to rank models whose intervals
overlap. This module is where that discipline lives.

Two estimators, chosen for the two kinds of metric the catalog distinguishes:

``wilson``
    For proportions (`refusal_rate`, `jailbreak_rate`, `apologize_rate`, ...). The normal
    approximation is badly behaved near 0 and 1 — exactly where safety metrics sit — and can
    produce intervals extending past the bounds. Wilson does not.

``bootstrap_mean``
    For a bounded mean over per-sample values (`strong_reject_metric`, 0-5). Requires the
    per-sample values from the log; when they are unavailable the record says so rather than
    inventing an interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# 95% two-sided normal quantile. Hardcoded rather than pulled from scipy so that the
# interval maths has no dependency and is trivially checkable against a table.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A confidence interval, or an explicit statement that none could be computed."""

    low: float
    high: float
    method: str
    level: float = 0.95

    @property
    def available(self) -> bool:
        return not (math.isnan(self.low) or math.isnan(self.high))

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap.

        Used for tie detection. Note this is a deliberately *conservative* test: two
        non-overlapping 95% intervals imply a difference, but overlapping ones do not imply
        the absence of one. The leaderboard therefore only ever claims 'tied', never
        'significantly different'.
        """
        if not (self.available and other.available):
            return True  # unknown uncertainty: refuse to claim a difference
        return self.low <= other.high and other.low <= self.high

    def as_dict(self) -> dict[str, float | str]:
        return {"ci_low": self.low, "ci_high": self.high, "ci_method": self.method,
                "ci_level": self.level}


UNAVAILABLE = Interval(math.nan, math.nan, "unavailable")


def wilson(successes: float, n: int, z: float = Z_95, scale: float = 1.0) -> Interval:
    """Wilson score interval for a proportion, returned on the metric's native scale.

    ``scale`` is the metric's upper bound: 1.0 for a fraction, 100.0 for a percentage. The
    proportion is computed in [0, 1] and the interval is scaled back afterwards, so a
    percentage metric gets a percentage interval.

    Args:
        successes: count of successes (may be fractional — XSTest counts a partial refusal
            as half a refusal before rounding into its rate).
        n: denominator. **This must be the number of *scored* samples**, not the number
            requested: samples the grader failed to parse leave the denominator entirely.
        z: normal quantile for the desired level.
        scale: metric upper bound.

    Returns:
        An ``Interval`` on the native scale, or ``UNAVAILABLE`` when ``n <= 0``.
    """
    if n <= 0:
        return UNAVAILABLE
    p = successes / n
    p = min(1.0, max(0.0, p))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(
        low=max(0.0, centre - margin) * scale,
        high=min(1.0, centre + margin) * scale,
        method=f"wilson-{'pct' if scale != 1.0 else 'unit'}",
    )


def wilson_from_rate(rate: float, n: int, scale: float = 1.0, z: float = Z_95) -> Interval:
    """Wilson interval when only the aggregate rate is known, not the raw counts.

    Inspect reports the metric, not the successes, so this back-computes the count. It is
    exact for integer-valued counts and a close approximation otherwise.
    """
    if n <= 0 or math.isnan(rate):
        return UNAVAILABLE
    return wilson(successes=(rate / scale) * n, n=n, z=z, scale=scale)


def bootstrap_mean(
    values: Sequence[float],
    iterations: int = 10_000,
    level: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap interval for the mean of per-sample values.

    Seeded, so the published interval is reproducible from the same log — an interval that
    moves between renderings of the same data is not evidence of anything.
    """
    clean = [v for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return UNAVAILABLE
    if n == 1:
        return Interval(clean[0], clean[0], "bootstrap-degenerate", level)

    import random

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - level) / 2.0
    lo = means[int(alpha * iterations)]
    hi = means[min(iterations - 1, int((1.0 - alpha) * iterations))]
    return Interval(low=lo, high=hi, method="bootstrap-percentile", level=level)


def propagate_complement(interval: Interval, upper: float) -> Interval:
    """Reflect an interval through ``upper - v``.

    The calibration chart plots compliance (100 - refusal_rate) and refusal
    (1 - jailbreak_rate); the error bars must be reflected with the point, and the bounds
    swap when they are.
    """
    if not interval.available:
        return UNAVAILABLE
    return Interval(low=upper - interval.high, high=upper - interval.low,
                    method=interval.method, level=interval.level)


def combine_weighted(
    values: Sequence[float], intervals: Sequence[Interval], weights: Sequence[float]
) -> tuple[float, Interval]:
    """Weighted mean of normalised metrics, with a deliberately conservative interval.

    The composite index adds up metrics that are neither independent nor identically scaled,
    so a proper variance calculation would be a fiction. Instead the half-widths are combined
    linearly, which over-states the uncertainty rather than under-stating it. An index whose
    interval is too wide loses a ranking claim; one that is too narrow makes a false one.
    """
    pairs = [
        (v, i, w)
        for v, i, w in zip(values, intervals, weights, strict=False)
        if not math.isnan(v)
    ]
    if not pairs:
        return math.nan, UNAVAILABLE
    total_w = sum(w for _, _, w in pairs)
    if total_w <= 0:
        return math.nan, UNAVAILABLE
    point = sum(v * w for v, _, w in pairs) / total_w
    half_widths = [
        ((i.high - i.low) / 2.0 if i.available else 0.0, w) for _, i, w in pairs
    ]
    if all(hw == 0.0 for hw, _ in half_widths):
        return point, UNAVAILABLE
    half = sum(hw * w for hw, w in half_widths) / total_w
    return point, Interval(
        low=max(0.0, point - half), high=min(1.0, point + half), method="weighted-linear"
    )
