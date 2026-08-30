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
from dataclasses import dataclass
from enum import Enum

from .config import GateSpec, RunConfig
from .results import CellStatus, ResultSet


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
    detail: str = ""

    @property
    def bound_text(self) -> str:
        suffix = "%" if self.unit == "percent" else ""
        if self.bound_min is not None and self.bound_max is not None:
            return f"{self.bound_min:g}{suffix} - {self.bound_max:g}{suffix}"
        if self.bound_max is not None:
            return f"<= {self.bound_max:g}{suffix}"
        return f">= {self.bound_min:g}{suffix}"

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
        breaches.append(f"{metric.value:.4g} > max {gate.max:g}")
    if gate.min is not None and metric.value < gate.min:
        breaches.append(f"{metric.value:.4g} < min {gate.min:g}")

    if breaches:
        return GateResult(**base, outcome=GateOutcome.FAIL, observed=metric.value,
                          detail="; ".join(breaches))

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
    return GateResult(**base, outcome=GateOutcome.PASS, observed=metric.value, detail=detail)


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
        "| gate | model | metric | bound | observed | outcome | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        mark = {GateOutcome.PASS: "pass", GateOutcome.FAIL: "**FAIL**",
                GateOutcome.ERROR: "_error_"}[r.outcome]
        lines.append(
            f"| `{r.gate_id}` | {r.model_label} | {r.metric_label} | {r.bound_text} | "
            f"{r.observed_text} | {mark} | {r.detail or ''} |"
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
