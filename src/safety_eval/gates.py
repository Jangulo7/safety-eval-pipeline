"""Release gates: thresholds that fail the build.

This is what makes "release gating" a fact rather than a claim. ``safety-eval gate`` exits 1
on any breach.

Two decisions worth stating:

**A missing metric is a gate error, not a pass.** If a cell errored, was blocked, or produced
``nan``, the gate cannot be evaluated. Treating that as a pass would mean a broken run
silently ships. It is reported as ``ERROR`` and fails the build by default.

**Bounds are checked against the catalog range at config load time**, not here. A gate of
``max: 0.20`` on XSTest — whose ``refusal_rate`` is a 0-100 percentage — can never fire, and
a gate that can never fire is worse than no gate because it looks like coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .config import GateSpec, RunConfig
from .results import CellStatus, ResultSet
from .stats import wilson_from_rate


class GateOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    """The gate could not be evaluated — errored cell, blocked dataset, or a nan metric."""


@dataclass(frozen=True)
class GateResult:
    """One gate evaluated against one model."""

    gate_id: str
    task_key: str
    metric_address: str
    metric_label: str
    model_id: str
    model_label: str
    outcome: GateOutcome
    observed: float
    bound_min: float | None
    bound_max: float | None
    unit: str | None
    rationale: str
    bound_min_stratum: float | None = None
    bound_max_stratum: float | None = None
    worst_stratum: str | None = None
    worst_stratum_value: float = math.nan
    failing_strata: list[tuple[str, float, int]] = field(default_factory=list)
    detail: str = ""

    @property
    def bound_text(self) -> str:
        suffix = "%" if self.unit == "percent" else ""
        parts = []
        if self.bound_min is not None and self.bound_max is not None:
            parts.append(f"{self.bound_min:g}{suffix} - {self.bound_max:g}{suffix}")
        elif self.bound_max is not None:
            parts.append(f"<= {self.bound_max:g}{suffix}")
        elif self.bound_min is not None:
            parts.append(f">= {self.bound_min:g}{suffix}")
        if self.bound_min_stratum is not None:
            parts.append(f"每 stratum >= {self.bound_min_stratum:g}{suffix}")
        if self.bound_max_stratum is not None:
            parts.append(f"每 stratum <= {self.bound_max_stratum:g}{suffix}")
        return "; ".join(parts).replace("每", "each")

    @property
    def stratum_text(self) -> str:
        """The worst category, which is what a per-stratum breach is actually about."""
        if self.worst_stratum is None or math.isnan(self.worst_stratum_value):
            return ""
        suffix = "%" if self.unit == "percent" else ""
        return f"{self.worst_stratum} {self.worst_stratum_value:.4g}{suffix}"

    @property
    def observed_text(self) -> str:
        if math.isnan(self.observed):
            return "n/a"
        suffix = "%" if self.unit == "percent" else ""
        return f"{self.observed:.4g}{suffix}"


@dataclass
class GateReport:
    """All gates against all models, plus the exit code that follows."""

    results: list[GateResult]
    fail_on_error: bool = True

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.outcome is GateOutcome.FAIL]

    @property
    def errors(self) -> list[GateResult]:
        return [r for r in self.results if r.outcome is GateOutcome.ERROR]

    @property
    def passed(self) -> bool:
        if self.failures:
            return False
        return not (self.errors and self.fail_on_error)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def by_model(self) -> dict[str, list[GateResult]]:
        out: dict[str, list[GateResult]] = {}
        for r in self.results:
            out.setdefault(r.model_id, []).append(r)
        return out

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.outcome is GateOutcome.PASS)
        return (
            f"{n_pass} passed, {len(self.failures)} failed, {len(self.errors)} "
            f"could not be evaluated"
        )


def evaluate(
    results: ResultSet, config: RunConfig, *, fail_on_error: bool = True
) -> GateReport:
    """Evaluate every configured gate against every model in the result set."""
    out: list[GateResult] = []
    for gate in config.gates:
        _, metric_spec = config.resolve_reference(f"{gate.task}:{gate.metric}")
        for model_id in results.models:
            out.append(
                _evaluate_one(gate, metric_spec, results, model_id)
            )
    return GateReport(out, fail_on_error=fail_on_error)


def _evaluate_one(
    gate: GateSpec, metric_spec, results: ResultSet, model_id: str
) -> GateResult:
    cell = results.get(gate.task, model_id)
    base = dict(  # noqa: C408 — kwargs form keeps this readable against GateResult's fields
        gate_id=gate.id,
        task_key=gate.task,
        metric_address=metric_spec.address,
        metric_label=metric_spec.label,
        model_id=model_id,
        model_label=results.label_for(model_id),
        bound_min=gate.min,
        bound_max=gate.max,
        bound_min_stratum=gate.min_per_stratum,
        bound_max_stratum=gate.max_per_stratum,
        unit=metric_spec.unit,
        rationale=gate.rationale,
    )

    if cell is None:
        return GateResult(**base, outcome=GateOutcome.ERROR, observed=math.nan,
                          detail="cell was not run")
    if cell.status is not CellStatus.OK:
        return GateResult(**base, outcome=GateOutcome.ERROR, observed=math.nan,
                          detail=f"cell {cell.status.value}: {cell.error_message or ''}".strip())

    metric = cell.metric(metric_spec.address)
    if metric is None:
        return GateResult(**base, outcome=GateOutcome.ERROR, observed=math.nan,
                          detail=f"metric {metric_spec.address} absent from the log")
    if metric.is_nan:
        return GateResult(
            **base, outcome=GateOutcome.ERROR, observed=math.nan,
            detail=f"metric is nan — {metric.unscored_samples} of "
                   f"{metric.scored_samples + metric.unscored_samples} samples were unscored, "
                   "so the grader, not the model, is what failed",
        )

    breaches: list[str] = []
    if gate.max is not None and metric.value > gate.max:
        breaches.append(f"aggregate {metric.value:.4g} > max {gate.max:g}")
    if gate.min is not None and metric.value < gate.min:
        breaches.append(f"aggregate {metric.value:.4g} < min {gate.min:g}")

    # Per-stratum bounds. An aggregate can be cleared while one category fails badly, and a
    # gate a model passes with a category that bad is measuring the wrong thing.
    failing, worst, worst_value = _stratum_breaches(gate, metric)
    if failing:
        described = []
        for s, v, n in failing[:4]:
            decisive = stratum_is_decisive(gate, metric, v, n)
            mark = {True: "", False: ", NOT decisive at this n",
                    None: ", no interval available for this metric"}[decisive]
            described.append(f"{s} {v:.4g} (n={n}{mark})")
        more = f" and {len(failing) - 4} more" if len(failing) > 4 else ""
        n_soft = sum(1 for s, v, n in failing
                     if stratum_is_decisive(gate, metric, v, n) is False)
        n_unknown = sum(1 for s, v, n in failing
                        if stratum_is_decisive(gate, metric, v, n) is None)
        if n_soft:
            qualifier = f" — {n_soft} of them not decisive at this sample size"
        elif n_unknown == len(failing):
            qualifier = (" — decisiveness not assessed: this metric is a bounded mean, not "
                         "a proportion, so no per-stratum interval is computed for it")
        else:
            qualifier = " — every assessable one decisive at 95%"
        breaches.append(
            f"{len(failing)} of {len(metric.per_stratum)} strata breach the per-stratum "
            f"bound: {', '.join(described)}{more}{qualifier}"
        )

    if breaches:
        return GateResult(**base, outcome=GateOutcome.FAIL, observed=metric.value,
                          worst_stratum=worst, worst_stratum_value=worst_value,
                          failing_strata=failing, detail="; ".join(breaches))

    # A pass whose interval straddles the bound is not a comfortable pass, and the report
    # says so rather than presenting a coin-flip as compliance.
    detail = ""
    straddles = not math.isnan(metric.ci_low) and (
        (gate.max is not None and metric.ci_high > gate.max)
        or (gate.min is not None and metric.ci_low < gate.min)
    )
    if straddles:
        detail = ("passes on the point estimate, but the 95% interval "
                  f"[{metric.ci_low:.3g}, {metric.ci_high:.3g}] crosses the bound")
    return GateResult(**base, outcome=GateOutcome.PASS, observed=metric.value,
                      worst_stratum=worst, worst_stratum_value=worst_value, detail=detail)


def _stratum_breaches(
    gate: GateSpec, metric
) -> tuple[list[tuple[str, float, int]], str | None, float]:
    """Categories that breach a per-stratum bound, plus the worst one either way.

    Strata are small — 25 samples per XSTest prompt type — so a breach is reported with
    whether its 95% interval actually clears the bound. A point estimate 2 points past a
    threshold at n=25 is not evidence of anything, and a gate that treats it as evidence is
    manufacturing findings out of sampling noise.
    """
    per_stratum = getattr(metric, "per_stratum", None) or {}
    if not per_stratum:
        return [], None, math.nan

    lower_is_worse = gate.min_per_stratum is not None or gate.min is not None
    worst = (min if lower_is_worse else max)(per_stratum.items(),
                                             key=lambda kv: kv[1][0])
    failing = []
    for stratum, (value, n) in sorted(per_stratum.items(), key=lambda kv: kv[1][0],
                                      reverse=not lower_is_worse):
        if (gate.min_per_stratum is not None and value < gate.min_per_stratum) or (gate.max_per_stratum is not None and value > gate.max_per_stratum):
            failing.append((stratum, value, int(n)))
    return failing, worst[0], worst[1][0]


# Metrics that are genuinely proportions, and so admit a Wilson interval per stratum.
# Anything else — a bounded mean, a ratio over a per-model subset — does not.
_RATE_METRICS = {
    "model_graded_qa/refusal_rate",
    "strong_reject_scorer/jailbreak_rate",
}


def stratum_is_decisive(gate: GateSpec, metric, value: float, n: int) -> bool | None:
    """Whether a stratum's whole 95% interval clears the bound it breached.

    Strata are small -- 25 samples per XSTest prompt type -- so a point estimate two points
    past a threshold is not evidence of anything. A gate that treats it as evidence is
    manufacturing findings out of sampling noise. Returns ``None`` when no interval can be
    formed.
    """
    if n <= 0 or metric.range_high <= metric.range_low:
        return None
    # Wilson is an interval for a PROPORTION. `strong_reject_metric` is a bounded mean of
    # per-sample severity scores, not a proportion, so Wilson does not apply to it and
    # pretending otherwise would put a fabricated interval behind a gate decision. The
    # aggregate for such a metric is bootstrapped; the per-stratum values are not, so no
    # decisiveness claim is made for them at all.
    if metric.address not in _RATE_METRICS:
        return None
    interval = wilson_from_rate(value, n, scale=metric.range_high)
    if not interval.available:
        return None
    if gate.min_per_stratum is not None:
        return interval.high < gate.min_per_stratum
    if gate.max_per_stratum is not None:
        return interval.low > gate.max_per_stratum
    return None


def render_markdown(report: GateReport, results: ResultSet, config: RunConfig) -> str:
    """Render ``gate_report.md``."""
    lines = [
        "# Gate report",
        "",
        f"Run `{results.metadata.run_id}` · "
        f"{results.metadata.finished_utc or results.metadata.started_utc}",
        "",
        f"**{'PASS' if report.passed else 'FAIL'}** — {report.summary()}",
        "",
        "> Thresholds are illustrative defaults chosen to demonstrate the gating mechanism.",
        "> They are not safety claims and must not be cited as such.",
        "",
        "| gate | model | metric | bound | observed | worst stratum | outcome | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        mark = {GateOutcome.PASS: "pass", GateOutcome.FAIL: "**FAIL**",
                GateOutcome.ERROR: "_error_"}[r.outcome]
        lines.append(
            f"| `{r.gate_id}` | {r.model_label} | {r.metric_label} | {r.bound_text} | "
            f"{r.observed_text} | {r.stratum_text or '—'} | {mark} | {r.detail or ''} |"
        )

    if config.dropped_gates:
        lines += [
            "",
            "> **This run was scoped to a subset of the benchmarks.** "
            + f"{len(config.dropped_gates)} gate(s) — "
            + ", ".join(f"`{g}`" for g in config.dropped_gates)
            + " — were not evaluated because their task was not run. A pass here is not "
              "full coverage.",
        ]

    rationales = {g.id: g.rationale for g in config.gates}
    lines += ["", "## Why each gate exists", ""]
    for gate_id, rationale in rationales.items():
        lines.append(f"- **`{gate_id}`** — {rationale}")

    if report.errors:
        lines += [
            "",
            "## Gates that could not be evaluated",
            "",
            "A gate with no number is reported as an error and fails the build. Treating an "
            "unevaluable gate as a pass would let a broken run ship silently.",
            "",
        ]
        for r in report.errors:
            lines.append(f"- `{r.gate_id}` / {r.model_label}: {r.detail}")

    return "\n".join(lines) + "\n"
