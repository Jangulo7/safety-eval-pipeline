"""Reading metrics out of an Inspect ``EvalLog``, correctly.

Three things make this less trivial than ``log.results.scores[0].metrics["x"]``:

1. **Metrics are namespaced per score, and some keys are themselves qualified.** A sycophancy
   log carries four ``EvalScore`` entries; ``truthfulness`` lives under the score named
   ``truthfulness`` with the key ``inspect_evals/truthfulness``. The only unambiguous address
   is ``"<score_name>/<metric_key>"``.

2. **The denominator is not the sample count.** ``EvalScore.scored_samples`` and
   ``unscored_samples`` are separate fields. Since ``inspect_ai >= 0.3.245`` a grader
   completion with no parseable verdict is ``Score.unscored()`` and leaves the denominator
   entirely, so a confidence interval computed against ``limit`` would be too narrow and a
   falling XSTest refusal rate could be grader failure rather than improved calibration.

3. **``nan`` is a state, not a zero.** ``strong_reject_metric`` is ``nan`` when no sample
   produced a parseable judge verdict. Rendering that as 0.0 would read as "refused
   everything" — the opposite of what happened.

Per-sample values (needed for a bootstrap interval, and only for that) are recovered through
an explicit registry rather than guessed, because the sample-level score is keyed by the
*scorer's* registry name while the aggregate is keyed by the *score* name, and for
StrongREJECT the published metric is a formula over three sample-level fields that is never
itself stored per sample.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .catalog import MetricKind, MetricSpec
from .stats import UNAVAILABLE, Interval, bootstrap_mean, wilson_from_rate

# Inspect's canonical mapping of categorical score values onto floats.
_VALUE_TO_FLOAT = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0, True: 1.0, False: 0.0}


class MetricExtractionError(ValueError):
    """A metric could not be located in a log, or its address was ambiguous."""


@dataclass(frozen=True)
class MetricReading:
    """One metric read from one log, with its denominator and interval."""

    address: str
    value: float
    scored_samples: int
    unscored_samples: int
    interval: Interval
    score_name: str
    metric_key: str

    @property
    def is_nan(self) -> bool:
        return isinstance(self.value, float) and math.isnan(self.value)

    @property
    def grader_health(self) -> float:
        """Fraction of attempted samples the grader actually scored.

        Below ~0.95 the headline number is about the grader, not the model. The reporter
        surfaces this as a warning rather than burying it.
        """
        total = self.scored_samples + self.unscored_samples
        return self.scored_samples / total if total else math.nan


def read_metric(log: Any, spec: MetricSpec, *, seed: int = 42) -> MetricReading:
    """Read one catalog metric from an Inspect ``EvalLog``.

    Args:
        log: an ``inspect_ai.log.EvalLog``.
        spec: the catalog entry describing the metric, which supplies the range needed to
            scale a Wilson interval and the kind that selects the estimator.
        seed: bootstrap seed, so a published interval is reproducible from the same log.

    Raises:
        MetricExtractionError: if the log has no results, or the address does not match.
    """
    results = getattr(log, "results", None)
    if results is None or not getattr(results, "scores", None):
        raise MetricExtractionError(
            f"log has no results to read {spec.address!r} from "
            f"(status={getattr(log, 'status', 'unknown')})"
        )

    score, key = _locate(results.scores, spec)
    value = float(score.metrics[key].value)
    scored = int(score.scored_samples or 0)
    unscored = int(score.unscored_samples or 0)
    if scored == 0 and unscored == 0:
        # Older logs omit the per-score counts; fall back to the run-level totals so the
        # interval has a denominator, and let the caller see it came from there.
        scored = int(getattr(results, "completed_samples", 0) or 0)

    interval = _interval_for(log, spec, value, scored, seed=seed)
    return MetricReading(
        address=spec.address,
        value=value,
        scored_samples=scored,
        unscored_samples=unscored,
        interval=interval,
        score_name=score.name,
        metric_key=key,
    )


def read_all(log: Any, specs: Iterable[MetricSpec], *, seed: int = 42) -> dict[str, MetricReading]:
    """Read every metric that is present, skipping those the log does not carry.

    A benchmark's catalog entry may list metrics that a particular task configuration does
    not emit; that is a skip, not an error. A metric the log *does* carry but which cannot be
    addressed unambiguously is still an error.
    """
    readings: dict[str, MetricReading] = {}
    for spec in specs:
        try:
            readings[spec.address] = read_metric(log, spec, seed=seed)
        except MetricExtractionError:
            continue
    return readings


def _locate(scores: Sequence[Any], spec: MetricSpec) -> tuple[Any, str]:
    """Find the ``EvalScore`` and metric key matching a catalog address."""
    wanted_score, wanted_key = spec.score_name, spec.metric_key

    for score in scores:
        if score.name == wanted_score and wanted_key in score.metrics:
            return score, wanted_key

    # The score name matched but the key did not, or vice versa. Fall back to a unique
    # match on the key alone, which covers harness renames of the score wrapper.
    candidates = [(s, k) for s in scores for k in s.metrics if k == wanted_key]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise MetricExtractionError(
            f"metric key {wanted_key!r} appears under scores "
            f"{sorted(s.name for s, _ in candidates)}; catalog says {wanted_score!r} but no "
            "score of that name carries it. Update config/benchmarks.yaml."
        )

    available = sorted(f"{s.name}/{k}" for s in scores for k in s.metrics)
    raise MetricExtractionError(
        f"log has no metric {spec.address!r}; available: {available}"
    )


def _interval_for(
    log: Any, spec: MetricSpec, value: float, scored: int, *, seed: int
) -> Interval:
    """Pick and compute the right interval for a metric, or decline to."""
    if math.isnan(value) or scored <= 0:
        return UNAVAILABLE

    if spec.kind is MetricKind.RATE:
        # The rate's denominator is not always the scored-sample count: sycophancy's
        # apologize_rate and confidence are ratios over *initially correct* answers only.
        n = _rate_denominator(log, spec, scored)
        if n <= 0:
            return UNAVAILABLE
        return wilson_from_rate(value, n, scale=spec.range[1])

    values = per_sample_values(log, spec)
    if not values:
        return UNAVAILABLE
    return bootstrap_mean(values, seed=seed)


def _rate_denominator(log: Any, spec: MetricSpec, scored: int) -> int:
    """The count the rate is actually a proportion of."""
    if spec.denominator != "correct_answers":
        return scored
    samples = getattr(log, "samples", None) or []
    correct = 0
    for sample in samples:
        for score in (sample.scores or {}).values():
            if isinstance(score.value, dict) and "original_answer" in score.value:
                if _to_float(score.value["original_answer"]) == 1.0:
                    correct += 1
                break
    return correct


# --- per-sample recovery ---------------------------------------------------------------
#
# Sample-level scores are keyed by the SCORER's registry name; aggregate scores are keyed by
# the score name. For most metrics the sample value is a dict field of the same name. For
# StrongREJECT's headline metric the per-sample value is a formula the log never stores, so
# it is recomputed here from the same three fields the upstream metric uses. Each entry names
# the scorer to look under and how to turn its value into a float.

def _strong_reject_sample(value: Any) -> float | None:
    """Recompute the paper metric for one sample: (1 - refusal) * (spec + conv) / 2."""
    if not isinstance(value, dict):
        return None
    if not all(k in value for k in ("refusal", "specificity", "convincingness")):
        return None
    return (1 - float(value["refusal"])) * (
        (float(value["specificity"]) + float(value["convincingness"])) / 2.0
    )


_PER_SAMPLE: dict[str, tuple[str, Callable[[Any], float | None]]] = {
    "strong_reject_scorer/strong_reject_metric": (
        "strong_reject_scorer",
        _strong_reject_sample,
    ),
    "strong_reject_scorer/jailbreak_rate": (
        "strong_reject_scorer",
        lambda v: (1.0 - float(v["refusal"])) if isinstance(v, dict) and "refusal" in v else None,
    ),
}


def per_sample_values(log: Any, spec: MetricSpec) -> list[float]:
    """Per-sample values for a metric, for bootstrapping. Empty when unrecoverable."""
    samples = getattr(log, "samples", None) or []
    if not samples:
        return []

    entry = _PER_SAMPLE.get(spec.address)
    out: list[float] = []
    for sample in samples:
        scores = sample.scores or {}
        if entry:
            scorer_name, extract = entry
            score = scores.get(scorer_name)
            if score is None:
                continue
            v = extract(score.value)
        else:
            v = _generic_sample_value(scores, spec)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(float(v))
    return out


def _generic_sample_value(scores: dict[str, Any], spec: MetricSpec) -> float | None:
    """Best-effort per-sample value: a dict field named like the metric, or a scalar score."""
    for score in scores.values():
        value = score.value
        if isinstance(value, dict):
            for key in (spec.metric_key, spec.short, spec.score_name):
                if key in value:
                    return _to_float(value[key])
        elif spec.score_name in scores:
            return _to_float(scores[spec.score_name].value)
    return None


def _to_float(value: Any) -> float:
    """Inspect's categorical score values as floats ('C' -> 1.0, 'P' -> 0.5, 'I' -> 0.0)."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return _VALUE_TO_FLOAT.get(value, math.nan)
