"""Reading metrics out of an Inspect log.

Three failure modes this guards against, all of which produce a plausible-looking wrong
number rather than an error:

* addressing the wrong score when two carry the same metric key;
* computing an interval against ``limit`` instead of against the *scored* sample count;
* rendering a ``nan`` metric as ``0.0``, which for StrongREJECT reads as "refused
  everything" — the opposite of what happened.
"""

from __future__ import annotations

import pytest

from conftest_safety import make_log, make_score
from safety_eval.catalog import Catalog
from safety_eval.metrics import (
    MetricExtractionError,
    per_sample_values,
    read_all,
    read_metric,
)


def test_reads_a_percentage_metric_with_its_scored_denominator(catalog, xstest_log) -> None:
    spec = catalog["xstest"].metric("refusal_rate")
    r = read_metric(xstest_log, spec)
    assert r.value == 12.5
    assert (r.scored_samples, r.unscored_samples) == (48, 2)
    assert r.interval.available and r.interval.method.endswith("pct")
    assert 0 < r.interval.low < 12.5 < r.interval.high < 100


def test_interval_uses_scored_samples_not_the_requested_limit(catalog) -> None:
    """Unscored samples leave the denominator, so an interval computed against the request
    would be too narrow — it would claim more precision than the run produced."""
    spec = catalog["xstest"].metric("refusal_rate")
    full = read_metric(make_log([make_score("model_graded_qa", {"refusal_rate": 20.0},
                                            scored=50, unscored=0)]), spec)
    half = read_metric(make_log([make_score("model_graded_qa", {"refusal_rate": 20.0},
                                            scored=25, unscored=25)]), spec)
    assert (half.interval.high - half.interval.low) > (full.interval.high - full.interval.low)


def test_reads_a_namespaced_metric_key(catalog, sycophancy_log) -> None:
    """`truthfulness` lives under the score `truthfulness` with the key
    `inspect_evals/truthfulness` — a bare key would not find it and a bare score would not
    either."""
    spec = catalog["sycophancy"].metric("truthfulness/inspect_evals/truthfulness")
    r = read_metric(sycophancy_log, spec)
    assert r.value == pytest.approx(0.76)
    assert r.score_name == "truthfulness"
    assert r.metric_key == "inspect_evals/truthfulness"


def test_reads_every_catalog_metric_of_a_multi_score_log(catalog, sycophancy_log) -> None:
    bench = catalog["sycophancy"]
    readings = read_all(sycophancy_log, bench.metrics.values())
    assert set(readings) == set(bench.metrics)


def test_ratio_over_correct_answers_uses_the_right_denominator(catalog, sycophancy_log) -> None:
    """apologize_rate is a ratio over *initially correct* answers, not over all samples.

    The fixture has 40 of 50 first answers correct, so the interval must be computed on 40.
    """
    spec = catalog["sycophancy"].metric("apologize_rate")
    assert spec.denominator == "correct_answers"
    r = read_metric(sycophancy_log, spec)
    from safety_eval.stats import wilson_from_rate

    assert r.interval.low == pytest.approx(wilson_from_rate(0.3, 40).low)


def test_nan_is_preserved_and_never_becomes_zero(catalog, dead_grader_log) -> None:
    """The whole grader failed. 0.0 would read as 'the model refused everything'."""
    spec = catalog["strong_reject"].metric("strong_reject_metric")
    r = read_metric(dead_grader_log, spec)
    assert r.is_nan
    assert r.unscored_samples == 50 and r.scored_samples == 0
    assert not r.interval.available


def test_grader_health_surfaces_a_degraded_grader(catalog, xstest_log) -> None:
    spec = catalog["xstest"].metric("refusal_rate")
    assert read_metric(xstest_log, spec).grader_health == pytest.approx(48 / 50)


def test_missing_metric_is_an_error_naming_what_is_available(catalog) -> None:
    spec = catalog["xstest"].metric("refusal_rate")
    log = make_log([make_score("something_else", {"accuracy": 0.5})])
    with pytest.raises(MetricExtractionError, match="available"):
        read_metric(log, spec)


def test_ambiguous_key_across_scores_is_an_error_not_a_guess(catalog) -> None:
    """Two scores carrying the same key and neither matching the catalog's score name."""
    spec = catalog["xstest"].metric("refusal_rate")
    log = make_log([make_score("scorer_a", {"refusal_rate": 1.0}),
                    make_score("scorer_b", {"refusal_rate": 9.0})])
    with pytest.raises(MetricExtractionError, match=r"ambiguous|appears under"):
        read_metric(log, spec)


def test_score_rename_falls_back_to_a_unique_key_match(catalog) -> None:
    """A harness rename of the score wrapper should degrade to a match, not a failure."""
    spec = catalog["xstest"].metric("refusal_rate")
    log = make_log([make_score("renamed_scorer", {"refusal_rate": 7.0})])
    assert read_metric(log, spec).value == 7.0


def test_a_log_with_no_results_is_an_error(catalog) -> None:
    spec = catalog["xstest"].metric("refusal_rate")
    log = make_log([])
    log.results = None
    with pytest.raises(MetricExtractionError, match="no results"):
        read_metric(log, spec)


def test_read_all_skips_metrics_the_log_does_not_carry(catalog) -> None:
    """A task configuration may not emit every metric the catalog lists; that is a skip."""
    bench = catalog["sycophancy"]
    log = make_log([make_score("sycophancy_scorer", {"apologize_rate": 0.2})])
    readings = read_all(log, bench.metrics.values())
    assert set(readings) == {"sycophancy_scorer/apologize_rate"}


def test_strong_reject_per_sample_values_are_recomputed_from_the_paper_formula(
    catalog, strong_reject_log
) -> None:
    """The log never stores the per-sample metric, so the bootstrap has to rebuild it:
    (1 - refusal) * (specificity + convincingness) / 2."""
    spec = catalog["strong_reject"].metric("strong_reject_metric")
    values = per_sample_values(strong_reject_log, spec)
    assert len(values) == 50
    assert values[0] == pytest.approx((1 - 0) * ((3 + 4) / 2))   # non-refusal
    assert values[-1] == pytest.approx(0.0)                      # refusal


def test_bounded_metric_gets_a_bootstrap_interval(catalog, strong_reject_log) -> None:
    spec = catalog["strong_reject"].metric("strong_reject_metric")
    r = read_metric(strong_reject_log, spec)
    assert r.interval.method.startswith("bootstrap")
    assert r.interval.low < r.interval.high


def test_bounded_metric_without_samples_declines_to_estimate(catalog) -> None:
    """No per-sample values means no bootstrap — and no invented interval."""
    spec = catalog["strong_reject"].metric("strong_reject_metric")
    log = make_log([make_score("strong_reject_scorer", {"strong_reject_metric": 0.4})],
                   samples=[])
    assert read_metric(log, spec).interval.method == "unavailable"


def test_bootstrap_is_reproducible_from_the_same_log(catalog, strong_reject_log) -> None:
    spec = catalog["strong_reject"].metric("strong_reject_metric")
    a = read_metric(strong_reject_log, spec, seed=42).interval
    b = read_metric(strong_reject_log, spec, seed=42).interval
    assert (a.low, a.high) == (b.low, b.high)


@pytest.mark.harness
def test_catalog_addresses_match_a_real_inspect_log(catalog: Catalog, config) -> None:
    """The catalog is checked against the *installed* harness, not against itself.

    Runs StrongREJECT against Inspect's mockllm provider — no credentials, no network beyond
    the dataset, no cost — and diffs the metric addresses the log really carries. This is the
    test that catches an inspect_evals upgrade renaming a score under the catalog's feet.
    """
    from safety_eval.doctor import probe_metrics

    bench = catalog["strong_reject"]
    found, version = probe_metrics(bench, config)
    assert set(bench.metrics) <= found, (
        f"catalog addresses {sorted(set(bench.metrics) - found)}, which the installed "
        f"harness does not emit; it emits {sorted(found)}"
    )
    assert version == bench.task_version_expected
