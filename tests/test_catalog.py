"""The benchmark catalog is the single source of truth for what a metric means.

If it is wrong, a gate fires on the wrong scale, a chart axis lies, and the PDF explains a
number that is not the number in the table. These tests hold it to that responsibility, and
the last two check it against the *installed* Inspect harness rather than against itself.
"""

from __future__ import annotations

import math

import pytest

from safety_eval.catalog import Catalog, CatalogError, Direction, MetricKind


def test_every_metric_has_range_direction_and_explanation(catalog: Catalog) -> None:
    for bench in catalog:
        assert bench.metrics, f"{bench.key} declares no metrics"
        for metric in bench.metrics.values():
            lo, hi = metric.range
            assert hi > lo, f"{metric.address} has a degenerate range"
            assert isinstance(metric.direction, Direction)
            assert isinstance(metric.kind, MetricKind)
            assert len(metric.explain) > 40, (
                f"{metric.address} needs a real explanation — it is rendered verbatim in the "
                "PDF and the dashboard"
            )


def test_every_benchmark_explains_what_it_does_not_measure(catalog: Catalog) -> None:
    """The 'does not measure' field is what stops a reader over-claiming from one number."""
    for bench in catalog:
        for key in ("measures", "why_it_matters", "does_not_measure", "reading_the_result"):
            assert bench.interpretation.get(key), f"{bench.key} is missing {key!r}"


def test_metric_addresses_are_score_qualified(catalog: Catalog) -> None:
    """Inspect namespaces metrics per score, so a bare key is ambiguous by construction."""
    for bench in catalog:
        for address in bench.metrics:
            assert "/" in address, f"{bench.key}: {address!r} is not '<score>/<metric>'"


def test_context_dependent_metric_resolves_per_subset(catalog: Catalog) -> None:
    """XSTest's refusal_rate is good when low on the safe subset and high on the unsafe one."""
    metric = catalog["xstest"].metric("refusal_rate")
    assert metric.direction is Direction.CONTEXT_DEPENDENT
    assert metric.direction_for("safe") is Direction.LOWER_BETTER
    assert metric.direction_for("unsafe") is Direction.HIGHER_BETTER
    with pytest.raises(CatalogError, match="needs a subset"):
        metric.direction_for(None)


def test_xstest_refusal_rate_is_a_percentage(catalog: Catalog) -> None:
    """The single most expensive thing to get wrong: a 0-1 gate on a 0-100 metric never fires."""
    metric = catalog["xstest"].metric("refusal_rate")
    assert metric.range == (0.0, 100.0)
    assert metric.unit == "percent"


def test_strong_reject_metric_is_zero_to_five(catalog: Catalog) -> None:
    metric = catalog["strong_reject"].metric("strong_reject_metric")
    assert metric.range == (0.0, 5.0)
    assert metric.kind is MetricKind.BOUNDED, "a 0-5 mean needs a bootstrap, not a Wilson"


def test_normalisation_puts_good_at_one(catalog: Catalog) -> None:
    xstest = catalog["xstest"].metric("refusal_rate")
    assert xstest.normalise(0.0, "safe") == pytest.approx(1.0)
    assert xstest.normalise(100.0, "safe") == pytest.approx(0.0)
    assert xstest.normalise(100.0, "unsafe") == pytest.approx(1.0)

    sr = catalog["strong_reject"].metric("strong_reject_metric")
    assert sr.normalise(0.0) == pytest.approx(1.0)
    assert sr.normalise(5.0) == pytest.approx(0.0)
    assert sr.normalise(2.5) == pytest.approx(0.5)


def test_normalisation_clamps_out_of_range_values(catalog: Catalog) -> None:
    """A harness that returns 105% must not produce a negative normalised score."""
    metric = catalog["xstest"].metric("refusal_rate")
    assert metric.normalise(105.0, "safe") == pytest.approx(0.0)
    assert metric.normalise(-3.0, "safe") == pytest.approx(1.0)


def test_normalisation_propagates_nan(catalog: Catalog) -> None:
    """nan means the grader failed. Turning it into 0.0 would read as 'refused everything'."""
    metric = catalog["strong_reject"].metric("strong_reject_metric")
    assert math.isnan(metric.normalise(math.nan))


def test_ambiguous_bare_key_is_an_error(catalog: Catalog) -> None:
    bench = catalog["sycophancy"]
    with pytest.raises(CatalogError, match="no metric"):
        bench.metric("definitely_not_a_metric")


def test_strong_reject_withholds_transcripts(catalog: Catalog) -> None:
    """Non-negotiable: its logs contain model responses to forbidden prompts."""
    assert catalog["strong_reject"].publish_logs is False
    assert catalog["xstest"].publish_logs is True
    assert catalog["sycophancy"].publish_logs is True


def test_malformed_catalog_is_rejected(tmp_path) -> None:
    import yaml

    bad = {"benchmarks": {"x": {"task": "t", "title": "T", "grader_kwarg": "g",
                                "interpretation": {}, "metrics": {"bare_key": {
                                    "label": "L", "range": [0, 1], "direction": "lower_better",
                                    "kind": "rate", "explain": "x" * 50}}}}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(CatalogError, match="score_name"):
        Catalog.load(path)


def test_context_dependent_without_subsets_is_rejected(tmp_path) -> None:
    """A direction that can never be resolved is a latent crash, caught at load time."""
    import yaml

    bad = {"benchmarks": {"x": {"task": "t", "title": "T", "grader_kwarg": "g",
                                "interpretation": {}, "metrics": {"s/m": {
                                    "label": "L", "range": [0, 1],
                                    "direction": "context_dependent",
                                    "kind": "rate", "explain": "x" * 50}}}}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(CatalogError, match="direction_by_subset"):
        Catalog.load(path)


# --- checks against the installed harness ------------------------------------------------

def test_grader_kwarg_matches_the_real_inspect_signature(catalog: Catalog) -> None:
    """`strong_reject` takes `judge_llm`; the others take `scorer_model`.

    Passing the wrong one is silently ignored by Inspect: the task grades with its default
    model instead, and the numbers are quietly ungoverned. This test is the reason the
    catalog records the kwarg per benchmark rather than assuming one name.
    """
    import importlib
    import inspect as pyinspect

    for bench in catalog:
        func_name = bench.task.split("/", 1)[1]
        module = importlib.import_module(f"inspect_evals.{func_name}")
        func = getattr(module, func_name)
        target = getattr(func, "__wrapped__", func)
        params = pyinspect.signature(target).parameters
        assert bench.grader_kwarg in params, (
            f"catalog says {bench.key} takes its grader as {bench.grader_kwarg!r}, but "
            f"inspect_evals.{func_name} accepts {sorted(params)}"
        )


def test_catalog_records_the_harness_it_was_verified_against(catalog: Catalog) -> None:
    """A metric registry with no provenance is a guess. This is what `doctor` diffs."""
    assert catalog.verified_against.get("inspect_ai")
    assert catalog.verified_against.get("inspect_evals")
    assert catalog.verified_against.get("date")
