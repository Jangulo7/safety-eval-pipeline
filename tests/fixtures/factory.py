"""Builds a realistic :class:`ResultSet` without touching a provider.

The shape deliberately includes the awkward cases, because those are the ones renderers get
wrong: a blocked cell (gated dataset), an errored cell (rate limit), a ``nan`` metric with a
full unscored count (dead grader), and a cell whose grader only scored 80% of samples.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from safety_eval.catalog import Catalog
from safety_eval.results import (
    CellResult,
    CellStatus,
    MetricResult,
    ResultSet,
    RunMetadata,
)
from safety_eval.stats import wilson_from_rate

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE — invented numbers, never publish"

MODELS = [
    ("openrouter/anthropic/claude-sonnet-4.5", "anthropic", "Claude Sonnet 4.5"),
    ("openrouter/openai/gpt-4.1", "openai", "GPT-4.1"),
    ("openrouter/meta-llama/llama-3.3-70b-instruct", "meta", "Llama 3.3 70B"),
]

# (task_key, metric_address) -> per-model value. Chosen so that the calibration trade-off is
# visible (one over-cautious model, one permissive, one balanced) and so that the default
# fixture is a clean gate PASS — tests that are about breaches introduce them explicitly,
# which keeps it obvious what each test is actually asserting.
VALUES: dict[tuple[str, str], list[float]] = {
    ("xstest_safe", "model_graded_qa/refusal_rate"): [8.0, 26.0, 4.0],
    ("xstest_unsafe", "model_graded_qa/refusal_rate"): [96.0, 98.0, 72.0],
    ("strong_reject", "strong_reject_scorer/jailbreak_rate"): [0.06, 0.02, 0.30],
    ("strong_reject", "strong_reject_scorer/strong_reject_metric"): [0.35, 0.10, 1.60],
    ("sycophancy", "sycophancy_scorer/apologize_rate"): [0.22, 0.31, 0.42],
    ("sycophancy", "sycophancy_scorer/confidence"): [0.78, 0.69, 0.58],
    ("sycophancy", "truthfulness/inspect_evals/truthfulness"): [0.74, 0.66, 0.48],
    ("sycophancy", "original_answer/mean"): [0.86, 0.84, 0.71],
}

# A second value table where every model clears every configured gate. Tests that are about
# a *breach* start from this and introduce exactly one, so it is always obvious which change
# caused the failure. The default "spread" table above is the realistic one: at these
# thresholds two of its three models genuinely breach, which is what makes it useful for
# exercising the reporting of failures.
CLEAN_VALUES: dict[tuple[str, str], list[float]] = {
    ("xstest_safe", "model_graded_qa/refusal_rate"): [8.0, 14.0, 4.0],
    ("xstest_unsafe", "model_graded_qa/refusal_rate"): [96.0, 98.0, 88.0],
    ("strong_reject", "strong_reject_scorer/jailbreak_rate"): [0.06, 0.02, 0.18],
    ("strong_reject", "strong_reject_scorer/strong_reject_metric"): [0.35, 0.10, 0.80],
    ("sycophancy", "sycophancy_scorer/apologize_rate"): [0.22, 0.31, 0.42],
    ("sycophancy", "sycophancy_scorer/confidence"): [0.78, 0.69, 0.58],
    ("sycophancy", "truthfulness/inspect_evals/truthfulness"): [0.74, 0.66, 0.55],
    ("sycophancy", "original_answer/mean"): [0.86, 0.84, 0.71],
}

TASKS = [
    ("sycophancy", "sycophancy", None),
    ("xstest_safe", "xstest", "safe"),
    ("xstest_unsafe", "xstest", "unsafe"),
    ("strong_reject", "strong_reject", None),
]


def make_results(
    limit: int = 50,
    *,
    catalog: Catalog | None = None,
    with_blocked: bool = False,
    with_error: bool = False,
    with_dead_grader: bool = False,
    degraded_grader: bool = False,
    profile: str = "spread",
    run_id: str = "run-fixture",
) -> ResultSet:
    """Build a synthetic result set.

    Args:
        limit: samples requested per cell.
        with_blocked: mark both XSTest cells for the third model as ``blocked`` (gated
            dataset), so partial-coverage handling is exercised.
        with_error: mark the third model's ``sycophancy`` cell as a rate-limit error.
        with_dead_grader: give the third model's ``strong_reject`` cell ``nan`` metrics with
            every sample unscored, so nan-is-not-zero handling is exercised.
        degraded_grader: leave 20% of the first model's ``xstest_safe`` samples unscored, so
            the grader-health warning fires.
        profile: ``"spread"`` (default) is realistic and breaches two gates; ``"clean"``
            clears every configured gate, so a test about one breach introduces exactly one.
    """
    values = {"spread": VALUES, "clean": CLEAN_VALUES}[profile]
    catalog = catalog or Catalog.load()
    meta = RunMetadata(
        run_id=run_id,
        started_utc="2026-08-30T12:00:00+00:00",
        finished_utc="2026-08-30T12:41:00+00:00",
        provider="openrouter",
        grader_model="openrouter/openai/gpt-4.1-mini",
        limit=limit,
        inspect_ai_version="0.3.260",
        inspect_evals_version="0.18.0",
        pipeline_version="2.0.0",
        notes=SYNTHETIC_MARKER,
    )
    results = ResultSet(meta)

    for task_key, bench_key, subset in TASKS:
        bench = catalog[bench_key]
        for i, (model_id, family, label) in enumerate(MODELS):
            cell = CellResult(
                run_id=run_id,
                task_key=task_key,
                benchmark=bench_key,
                task=bench.task,
                task_args={"subset": subset} if subset else {},
                model_id=model_id,
                family=family,
                label=label,
                provider="openrouter",
                grader_model=meta.grader_model,
                temperature=0.0,
                seed=42,
                max_connections=8,
                n_requested=limit,
                n_completed=limit,
                task_version=int(bench.task_version_expected.split("-")[0]),
                full_task_version=bench.task_version_expected,
                inspect_ai_version="0.3.260",
                inspect_evals_version="0.18.0",
                wall_clock_s=120.0 + 11 * i,
                input_tokens=limit * 320,
                output_tokens=limit * 95,
                total_tokens=limit * 415,
                log_published=bench.publish_logs,
                log_path=f"logs/{task_key}/{family}/fixture.eval" if bench.publish_logs else None,
            )

            if with_blocked and bench_key == "xstest" and i == 2:
                cell.status = CellStatus.BLOCKED
                cell.error_message = (
                    "DatasetNotFoundError: Dataset 'walledai/XSTest' is a gated dataset — "
                    "request access and set HF_TOKEN"
                )
                results.add(cell)
                continue
            if with_error and task_key == "sycophancy" and i == 2:
                cell.status = CellStatus.ERROR
                cell.error_message = "APIStatusError: 429 rate limit exceeded (after 1 retry)"
                cell.attempts = 2
                results.add(cell)
                continue

            dead = with_dead_grader and task_key == "strong_reject" and i == 2
            for address, spec in bench.metrics.items():
                key = (task_key, address)
                if key not in values:
                    continue
                value = values[key][i]
                scored, unscored = limit, 0
                if dead:
                    value, scored, unscored = math.nan, 0, limit
                elif degraded_grader and task_key == "xstest_safe" and i == 0:
                    scored, unscored = int(limit * 0.8), limit - int(limit * 0.8)

                interval = wilson_from_rate(value, scored, scale=spec.range[1])
                cell.metrics.append(
                    MetricResult(
                        address=address,
                        label=spec.label,
                        value=value,
                        ci_low=interval.low,
                        ci_high=interval.high,
                        ci_method=interval.method,
                        scored_samples=scored,
                        unscored_samples=unscored,
                        range_low=spec.range[0],
                        range_high=spec.range[1],
                        direction=spec.direction_for(subset).value,
                        unit=spec.unit,
                        primary=spec.primary,
                        normalised=spec.normalise(value, subset),
                    )
                )
            results.add(cell)

    return results


def write(path: str | Path, **kwargs: Any) -> Path:
    """Convenience for regenerating the committed fixture file."""
    return make_results(**kwargs).save(path)
