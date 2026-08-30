"""The self-contained HTML leaderboard.

Self-contained means exactly that: charts are inlined as base64 data URIs and there is no
external stylesheet, font or script. The page can be emailed, committed, or opened from a
filesystem with no network, which is the only way an artefact like this survives contact with
a real review process.

Everything on the page is rendered from ``results.json`` plus the catalog. The catalog is
also what the gates and the composite index read, so the explanation of a metric on this
page cannot drift away from the threshold applied to it.
"""

from __future__ import annotations

import base64
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import RunConfig
from ..disclosure import contamination_disclosure, parameter_register
from ..gates import GateOutcome, GateReport
from ..leaderboard import Leaderboard
from ..results import CellStatus, ResultSet
from . import theme
from .conditions import COVERAGE_NOTE, PREAMBLE
from .conditions import build as build_conditions

TEMPLATE_DIR = Path(__file__).parent / "templates"

LIMITATIONS = [
    "<b>n is small.</b> Samples are capped per task per model for cost. This is not a "
    "full-benchmark result and should not be compared with published full-benchmark numbers.",
    "<b>No generation variance.</b> One sample per prompt at temperature 0, so the intervals "
    "reflect sampling of <em>prompts</em>, not of completions. Re-running the same "
    "configuration will not reproduce the spread shown here.",
    "<b>Grader dependence is not quantified.</b> Judge-graded metrics inherit the judge's "
    "biases. One grader model is used throughout so the bias is at least held constant "
    "across models, but its size is not measured here.",
    "<b>Thresholds are illustrative.</b> The gate bounds demonstrate the mechanism. They are "
    "not safety claims and must not be cited as such.",
    "<b>The composite index is a choice.</b> It is a weighted mean of normalised metrics. "
    "Different weights give a different ranking, and no weighting is objectively correct.",
    "<b>Scores are not capability-controlled.</b> A weaker model can look safer by being less "
    "able to comply. Sycophancy's first-answer accuracy is reported partly as a check on this.",
]


def render_leaderboard_html(
    results: ResultSet,
    config: RunConfig,
    board: Leaderboard,
    gate_report: GateReport,
    chart_paths: dict[str, Path] | None = None,
    *,
    embed_charts: bool = True,
) -> str:
    """Render the leaderboard page as a single self-contained HTML string."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("leaderboard.html.j2")
    meta = results.metadata
    order = {m: i for i, m in enumerate(results.models)}

    rows = []
    for row in board.rows:
        cells = {}
        for ref in board.metric_order:
            c = row.metrics.get(ref)
            if c is None or not c.available:
                status = c.status if c else "not run"
                cells[ref] = {
                    "available": False,
                    "status": status,
                    "pill_class": {"blocked": "blocked", "error": "err",
                                   "nan": "err"}.get(status, "err"),
                }
            else:
                cells[ref] = {
                    "available": True,
                    "value": c.format_value(),
                    "ci": c.format_ci(),
                    "unscored": c.unscored_samples or 0,
                }
        rows.append({
            "rank_text": row.rank_text,
            "label": row.label,
            "color": theme.series_color(order.get(row.model_id, 99)),
            "index_text": row.index_text,
            "ci": (f"[{row.interval.low:.3f}, {row.interval.high:.3f}]"
                   if row.interval.available else ""),
            "bar_pct": 0 if math.isnan(row.index) else round(row.index * 100, 1),
            "tied": bool(row.tied_with),
            "partial": row.partial,
            "cells": cells,
        })

    benchmarks = []
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        if any(b["key"] == bench.key for b in benchmarks):
            continue
        benchmarks.append({
            "key": bench.key,
            "title": bench.title,
            "task": bench.task,
            "publish_logs": bench.publish_logs,
            "measures": bench.interpretation.get("measures", ""),
            "why": bench.interpretation.get("why_it_matters", ""),
            "not_measured": bench.interpretation.get("does_not_measure", ""),
            "reading": bench.interpretation.get("reading_the_result", ""),
            "metrics": [
                {
                    "short": m.short,
                    "range": f"{m.range[0]:g}–{m.range[1]:g}"
                             f"{' %' if m.unit == 'percent' else ''}",
                    "direction": _direction_text(m),
                    "explain": m.explain,
                }
                for m in bench.metrics.values()
            ],
            "caveats": [c.text for c in bench.caveats],
        })

    gate_cls = {GateOutcome.PASS: "pass", GateOutcome.FAIL: "fail", GateOutcome.ERROR: "err"}
    gates = [{
        "id": g.gate_id, "model": g.model_label, "metric": g.metric_label,
        "bound": g.bound_text, "observed": g.observed_text,
        "outcome": g.outcome.value, "cls": gate_cls[g.outcome], "detail": g.detail,
    } for g in gate_report.results]

    status_cls = {"ok": "pass", "error": "fail", "blocked": "blocked", "skipped": "err"}
    provenance = [{
        "task": c.task_key, "model": c.label, "status": c.status.value,
        "cls": status_cls.get(c.status.value, "err"),
        "version": c.full_task_version or c.task_version or "—",
        "grader": (c.grader_model or "—").rsplit("/", 1)[-1],
        "n_req": c.n_requested, "n_done": c.n_completed,
        "tokens": f"{c.total_tokens:,}" if c.total_tokens else "—",
        "time": f"{c.wall_clock_s:.0f}s" if c.wall_clock_s else "—",
        "logs": "published" if c.log_published else "withheld",
    } for c in results]

    charts = []
    if chart_paths:
        for name, caption in _CHART_CAPTIONS.items():
            path = chart_paths.get(name)
            if path and Path(path).exists():
                charts.append({
                    "src": _data_uri(Path(path)) if embed_charts else Path(path).name,
                    "alt": caption["alt"],
                    "caption": caption["caption"],
                })

    cond = build_conditions(results, config)
    conditions = {
        "preamble": PREAMBLE,
        "coverage_note": COVERAGE_NOTE.replace("**", ""),
        "shared": cond.shared,
        "all_identical": cond.all_identical,
        "divergence": [
            f"{task_key} · {field_name}: {', '.join(values)}"
            for task_key, diverged in cond.divergence.items()
            for field_name, values in diverged.items()
        ],
        "under_covered": cond.under_covered,
        "tasks": [{
            "key": tc.task_key,
            "rows": tc.rows,
            "coverage": tc.coverage_text,
            "fully_covered": tc.fully_covered,
            "strata": ", ".join(f"{k} {v}" for k, v in tc.stratum_counts.items()),
        } for tc in cond.tasks],
    }

    has_strata = any(m.per_stratum for c in results for m in c.metrics)
    disc = contamination_disclosure(results, config, per_stratum=has_strata)
    disclosure = {
        "rows": [{"name": n, "code": c, "why": w} for n, c, w in disc.rows()],
        "f2_notes": disc.f2_notes,
        "headline": disc.headline,
    }
    register: list[dict[str, Any]] = []
    section = None
    for row in parameter_register(results, config):
        if row.section != section:
            section = row.section
            register.append({"section": section, "rows": []})
        register[-1]["rows"].append(
            {"parameter": row.parameter, "value": row.value, "status": row.status,
             "cls": {"recorded": "pass", "not applicable": "",
                     "undisclosed": "blocked", "missing": "fail"}[row.status]})

    strata_tables = []
    for task in config.tasks:
        rows_, columns = [], []
        for model_id in results.models:
            cell = results.get(task.key, model_id)
            if cell is None or not cell.ok:
                continue
            for m in cell.metrics:
                if not m.per_stratum:
                    continue
                columns = columns or sorted(m.per_stratum)
                suffix = "%" if m.unit == "percent" else ""
                # NB "cells", not "values": Jinja resolves a dict's .values to the method.
                rows_.append({"label": cell.label, "metric": m.label, "cells": [
                    (f"{m.per_stratum[s][0]:.3g}{suffix}", int(m.per_stratum[s][1]))
                    if s in m.per_stratum else ("—", 0) for s in columns]})
        if rows_:
            strata_tables.append({"task": task.key, "metric": rows_[0]["metric"],
                                  "columns": columns, "rows": rows_})

    warnings = _warnings(results, board)
    if not cond.all_identical:
        warnings.insert(0, "<b>Run conditions were not identical across models.</b> The "
                           "ranking below is not like-for-like — see Run conditions.")
    if cond.under_covered:
        warnings.append(
            "<b>Incomplete stratum coverage.</b> " + "; ".join(cond.under_covered)
            + ". Those cells scored the categories they reached, not the whole benchmark.")

    return template.render(
        index_name=board.index_name,
        run_id=meta.run_id,
        finished=_pretty_time(meta.finished_utc or meta.started_utc),
        inspect_ai_version=meta.inspect_ai_version or "—",
        inspect_evals_version=meta.inspect_evals_version or "—",
        pipeline_version=meta.pipeline_version or "—",
        grader_model=meta.grader_model or "—",
        limit=results.sample_size_note().replace('**', ''),
        temperature=next((c.temperature for c in results), 0.0),
        seed=next((c.seed for c in results), 0),
        n_models=len(results.models),
        n_tasks=len(results.task_keys),
        synthetic=meta.notes if meta.notes else "",
        models=[{"label": results.label_for(m), "color": theme.series_color(i)}
                for i, m in enumerate(results.models)],
        rows=rows,
        metric_order=board.metric_order,
        metric_labels=board.metric_labels,
        notes=board.notes,
        warnings=warnings,
        charts=charts,
        benchmarks=benchmarks,
        gates=gates,
        gate_passed=gate_report.passed,
        gate_summary=gate_report.summary(),
        provenance=provenance,
        conditions=conditions,
        disclosure=disclosure,
        register=register,
        strata_tables=strata_tables,
        limitations=LIMITATIONS,
        theme="light",
    )


_CHART_CAPTIONS = {
    "calibration": {
        "alt": "Scatter of compliance on safe prompts against refusal on forbidden prompts",
        "caption": "A single safety score hides a trade-off between over-refusal and "
                   "under-refusal. Top right is the only good corner.",
    },
    "leaderboard": {
        "alt": "Ranked composite safety index with confidence intervals",
        "caption": "The composite index with 95% intervals. Overlapping whiskers mean the "
                   "ordering between those models is not supported by this sample.",
    },
    "metric_grid": {
        "alt": "Small multiples of each metric in its native units",
        "caption": "Each metric in its own units and direction — the check on what the "
                   "composite index folded together.",
    },
    "coverage": {
        "alt": "Stacked bars of scored, unscored and unrun samples per cell",
        "caption": "Sample coverage. An amber band is a grader that could not parse its own "
                   "output; those samples left the metric's denominator.",
    },
}


def _direction_text(metric) -> str:
    from ..catalog import Direction

    if metric.direction is Direction.CONTEXT_DEPENDENT:
        parts = [f"{s}: {d.value.replace('_', ' ')}"
                 for s, d in metric.direction_by_subset.items()]
        return "; ".join(parts)
    return metric.direction.value.replace("_", " ")


def _warnings(results: ResultSet, board: Leaderboard) -> list[str]:
    """Banners that must be seen before the table, not after it."""
    out: list[str] = []
    blocked = [c for c in results if c.status is CellStatus.BLOCKED]
    if blocked:
        tasks = sorted({c.task_key for c in blocked})
        out.append(
            f"<b>{len(blocked)} cell(s) blocked before running</b> — {', '.join(tasks)}. "
            "This is a setup problem (a gated dataset or a missing credential), not model "
            "behaviour. Run <code>safety-eval doctor</code> for the fix."
        )
    errored = [c for c in results if c.status is CellStatus.ERROR]
    if errored:
        out.append(
            f"<b>{len(errored)} cell(s) failed to run.</b> The matrix below has gaps; a "
            "partial result with visible gaps is reported rather than a filled-in one."
        )
    degraded = [w for row in board.rows for w in row.warnings if "grader scored only" in w]
    if degraded:
        out.append(
            "<b>Grader degradation detected.</b> " + "; ".join(degraded[:3])
            + ". The headline values for those cells describe the grader as much as the model."
        )
    return out


def _data_uri(path: Path) -> str:
    """Inline a PNG so the page needs no companion files."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _pretty_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso
