"""Confidence intervals.

At n=50 a two-point difference between two models is noise, so every published number
carries an interval and the leaderboard refuses to rank models whose intervals overlap. If
these estimators are wrong, that whole discipline is decoration.

The Wilson expectations below are textbook values, checked independently rather than
recorded from this implementation's own output.
"""

from __future__ import annotations

import math

import pytest

from safety_eval.stats import (
    UNAVAILABLE,
    Interval,
    bootstrap_mean,
    combine_weighted,
    propagate_complement,
    wilson,
    wilson_from_rate,
)


@pytest.mark.parametrize("successes,n,low,high", [
    (0, 50, 0.0000, 0.0713),      # zero events: the normal approximation gives [0, 0]
    (50, 50, 0.9287, 1.0000),     # and all events: symmetric by construction
    (25, 50, 0.3664, 0.6336),
    (5, 100, 0.0216, 0.1118),
    (3, 20, 0.0524, 0.3604),
])
def test_wilson_matches_published_values(successes, n, low, high) -> None:
    interval = wilson(successes, n)
    assert interval.low == pytest.approx(low, abs=5e-4)
    assert interval.high == pytest.approx(high, abs=5e-4)


def test_wilson_never_leaves_the_unit_interval() -> None:
    """The reason Wilson is used at all: safety rates live at 0 and 1."""
    for successes, n in [(0, 5), (5, 5), (1, 3), (0, 1)]:
        i = wilson(successes, n)
        assert 0.0 <= i.low <= i.high <= 1.0


def test_wilson_narrows_with_n() -> None:
    wide = wilson(5, 10)
    narrow = wilson(50, 100)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_zero_denominator_declines_to_estimate() -> None:
    """No samples means no interval — not a zero-width one."""
    assert wilson(0, 0) is UNAVAILABLE
    assert not wilson(0, 0).available


def test_percentage_scale_returns_a_percentage_interval() -> None:
    """XSTest reports 0-100, so its interval must too."""
    i = wilson_from_rate(12.0, 50, scale=100.0)
    assert 0.0 < i.low < 12.0 < i.high < 100.0
    assert i.method.endswith("pct")
    unit = wilson_from_rate(0.12, 50, scale=1.0)
    assert i.low == pytest.approx(unit.low * 100, rel=1e-9)


def test_nan_rate_declines_to_estimate() -> None:
    assert not wilson_from_rate(math.nan, 50).available


def test_bootstrap_is_deterministic_under_a_seed() -> None:
    """An interval that moves between renderings of the same data is not evidence.

    Reproducibility is the property that matters here; the second assertion only checks that
    the seed is actually being used, and compares the whole interval because a percentile of
    coarse discrete data can land on the same endpoint under two different seeds.
    """
    values = [0.0, 0.4, 1.5, 3.0, 0.2, 4.5, 1.0, 0.9, 2.7, 3.8, 0.1, 1.9]
    a = bootstrap_mean(values, iterations=2000, seed=7)
    b = bootstrap_mean(values, iterations=2000, seed=7)
    assert (a.low, a.high) == (b.low, b.high)
    assert (bootstrap_mean(values, iterations=2000, seed=8).low,
            bootstrap_mean(values, iterations=2000, seed=8).high) != (a.low, a.high)


def test_bootstrap_brackets_the_sample_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    i = bootstrap_mean(values, iterations=3000, seed=1)
    assert i.low < sum(values) / len(values) < i.high


def test_bootstrap_handles_degenerate_inputs() -> None:
    assert not bootstrap_mean([]).available
    single = bootstrap_mean([2.5])
    assert single.low == single.high == 2.5
    assert single.method == "bootstrap-degenerate"


def test_bootstrap_ignores_nan_values() -> None:
    i = bootstrap_mean([1.0, math.nan, 1.0, 1.0], iterations=500, seed=3)
    assert i.low == i.high == pytest.approx(1.0)


def test_complement_reflects_and_swaps_the_bounds() -> None:
    """The calibration chart plots 100 - refusal; the error bars must reflect with it."""
    original = Interval(10.0, 30.0, "wilson-pct")
    flipped = propagate_complement(original, 100.0)
    assert (flipped.low, flipped.high) == (70.0, 90.0)


def test_overlap_is_symmetric_and_conservative() -> None:
    a, b = Interval(0.1, 0.5, "x"), Interval(0.4, 0.9, "x")
    assert a.overlaps(b) and b.overlaps(a)
    c = Interval(0.6, 0.9, "x")
    assert not Interval(0.1, 0.5, "x").overlaps(c)


def test_unknown_uncertainty_refuses_to_claim_a_difference() -> None:
    """Two models can only be called different when both intervals are known."""
    assert UNAVAILABLE.overlaps(Interval(0.1, 0.2, "x"))
    assert Interval(0.1, 0.2, "x").overlaps(UNAVAILABLE)


def test_weighted_combination_skips_missing_metrics() -> None:
    values = [0.8, math.nan, 0.6]
    intervals = [Interval(0.7, 0.9, "x"), UNAVAILABLE, Interval(0.5, 0.7, "x")]
    point, interval = combine_weighted(values, intervals, [0.5, 0.3, 0.2])
    # (0.8 * 0.5 + 0.6 * 0.2) / 0.7
    assert point == pytest.approx((0.8 * 0.5 + 0.6 * 0.2) / 0.7)
    assert interval.available


def test_weighted_combination_of_nothing_is_nan_not_zero() -> None:
    point, interval = combine_weighted([math.nan], [UNAVAILABLE], [1.0])
    assert math.isnan(point)
    assert not interval.available


def test_composite_interval_over_states_rather_than_under_states() -> None:
    """The metrics are neither independent nor identically scaled, so the combination is
    deliberately linear: an index whose interval is too wide loses a ranking claim, while one
    that is too narrow makes a false one."""
    intervals = [Interval(0.4, 0.8, "x"), Interval(0.4, 0.8, "x")]
    _, combined = combine_weighted([0.6, 0.6], intervals, [0.5, 0.5])
    half = (combined.high - combined.low) / 2
    quadrature = math.sqrt(2 * (0.2 * 0.5) ** 2)
    assert half > quadrature
