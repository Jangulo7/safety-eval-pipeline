"""Ranking models across benchmarks — carefully.

Combining a 0-100 percentage where low is good, a 0-5 score where low is good, and a 0-1 rate
where high is good into one number is only meaningful if every metric's range and direction
are known. That is what ``config/benchmarks.yaml`` is for, and the composite index here is
defined entirely in terms of it.

Three honesty constraints, all enforced in code rather than in prose:

1. **Weights are a choice, not a measurement.** They come from the config and are printed
   above every table that uses them.
2. **A model with a missing cell is not silently advantaged.** Its index is computed over the
   metrics it does have, its coverage is reported, and it is marked partial.
3. **Overlapping intervals mean a tie.** Two models separated by less than their combined
   uncertainty are reported as tied and share a rank. At n=50 this happens often, which is
   the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import RunConfig
from .results import CellStatus, ResultSet
from .stats import UNAVAILABLE, Interval, combine_weighted


@dataclass
class MetricCell:
    """One metric of one model as it appears in a leaderboard row."""

    reference: str
    label: str
    task_key: str
    value: float
    normalised: float
    ci_low: float
    ci_high: float
    unit: str | None
    direction: str
    scored_samples: int
    unscored_samples: int
    status: str

    @property
    def available(self) -> bool:
        return self.status == "ok" and not math.isnan(self.value)

    def format_value(self) -> str:
        if not self.available:
            return "—"
        suffix = "%" if self.unit == "percent" else ""
        return f"{self.value:.4g}{suffix}"

    def format_ci(self) -> str:
        if math.isnan(self.ci_low):
            return ""
        suffix = "%" if self.unit == "percent" else ""
        return f"[{self.ci_low:.3g}, {self.ci_high:.3g}]{suffix}"


@dataclass
class LeaderboardRow:
    """One model's line on the leaderboard."""

    model_id: str
    label: str
    family: str
    index: float
    interval: Interval
    rank: int = 0
    tied_with: list[str] = field(default_factory=list)
    metrics: dict[str, MetricCell] = field(default_factory=dict)
    coverage: float = 1.0
    """Fraction of the weighted metrics that actually produced a number."""

    warnings: list[str] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return self.coverage < 1.0

    @property
    def index_text(self) -> str:
        return "—" if math.isnan(self.index) else f"{self.index:.3f}"

    @property
    def rank_text(self) -> str:
        """Rank, with an equals sign when tied. `=1` reads as 'joint first'."""
        if math.isnan(self.index):
            return "—"
        return f"={self.rank}" if self.tied_with else str(self.rank)


@dataclass
class Leaderboard:
    """The ranked table plus everything a reader needs to distrust it appropriately."""

    rows: list[LeaderboardRow]
    index_name: str
    weights: dict[str, float]
    notes: list[str] = field(default_factory=list)
    metric_order: list[str] = field(default_factory=list)
    metric_labels: dict[str, str] = field(default_factory=dict)

    @property
    def any_ties(self) -> bool:
        return any(r.tied_with for r in self.rows)

    def top(self) -> LeaderboardRow | None:
        return self.rows[0] if self.rows else None


def build(results: ResultSet, config: RunConfig) -> Leaderboard:
    """Compute the composite index and rank the models."""
    weights = config.leaderboard.normalised_weights
    metric_order = list(weights)
    metric_labels: dict[str, str] = {}
    rows: list[LeaderboardRow] = []

    for model_id in results.models:
        cells: dict[str, MetricCell] = {}
        values: list[float] = []
        intervals: list[Interval] = []
        used_weights: list[float] = []
        warnings: list[str] = []

        for reference, weight in weights.items():
            task_key = reference.split(":", 1)[0]
            spec = config.resolve_reference(reference)[1]
            metric_labels[reference] = f"{task_key} · {spec.label}"

            cell = results.get(task_key, model_id)
            if cell is None:
                cells[reference] = _missing(reference, spec, task_key, "not run")
                continue
            if cell.status is not CellStatus.OK:
                cells[reference] = _missing(reference, spec, task_key, cell.status.value)
                if cell.status is CellStatus.BLOCKED:
                    warnings.append(f"{task_key}: blocked ({cell.error_message or 'setup'})")
                else:
                    warnings.append(f"{task_key}: {cell.status.value}")
                continue

            m = cell.metric(spec.address)
            if m is None or m.is_nan:
                cells[reference] = _missing(reference, spec, task_key, "nan")
                warnings.append(
                    f"{task_key}/{spec.short}: no value — "
                    f"{m.unscored_samples if m else '?'} samples unscored"
                )
                continue

            cells[reference] = MetricCell(
                reference=reference, label=spec.label, task_key=task_key, value=m.value,
                normalised=m.normalised, ci_low=m.ci_low, ci_high=m.ci_high, unit=m.unit,
                direction=m.direction, scored_samples=m.scored_samples,
                unscored_samples=m.unscored_samples, status="ok",
            )
            values.append(m.normalised)
            # Normalise the interval onto the same 0-1 good-is-high axis as the point, so
            # the index's uncertainty is in index units rather than a mix of native ones.
            intervals.append(_normalised_interval(m, spec))
            used_weights.append(weight)

            if m.unscored_samples and m.grader_health < 0.95:
                warnings.append(
                    f"{task_key}/{spec.short}: grader scored only "
                    f"{m.grader_health:.0%} of samples — read the value as a statement "
                    "about the grader as much as the model"
                )

        index, interval = combine_weighted(values, intervals, used_weights)
        coverage = sum(used_weights) if used_weights else 0.0
        cell_obj = results.get(results.task_keys[0], model_id) if results.task_keys else None
        rows.append(
            LeaderboardRow(
                model_id=model_id,
                label=results.label_for(model_id),
                family=cell_obj.family if cell_obj else "",
                index=index,
                interval=interval,
                metrics=cells,
                coverage=coverage,
                warnings=warnings,
            )
        )

    _rank(rows, tie_on_ci=config.leaderboard.tie_on_overlapping_ci)

    notes = _notes(results, config, rows)
    return Leaderboard(
        rows=rows,
        index_name=config.leaderboard.index_name,
        weights=weights,
        notes=notes,
        metric_order=metric_order,
        metric_labels=metric_labels,
    )


def _missing(reference: str, spec, task_key: str, status: str) -> MetricCell:
    return MetricCell(
        reference=reference, label=spec.label, task_key=task_key, value=math.nan,
        normalised=math.nan, ci_low=math.nan, ci_high=math.nan, unit=spec.unit,
        direction=spec.direction.value, scored_samples=0, unscored_samples=0, status=status,
    )


def _normalised_interval(metric, spec) -> Interval:
    """Map a native-scale interval onto the 0-1 good-is-high index axis."""
    if math.isnan(metric.ci_low):
        return UNAVAILABLE
    lo, hi = spec.range
    span = hi - lo
    if span == 0:
        return UNAVAILABLE
    a = (metric.ci_low - lo) / span
    b = (metric.ci_high - lo) / span
    if metric.direction == "lower_better":
        a, b = 1.0 - b, 1.0 - a
    return Interval(low=max(0.0, min(a, b)), high=min(1.0, max(a, b)), method="normalised")


def _rank(rows: list[LeaderboardRow], *, tie_on_ci: bool) -> None:
    """Sort by index descending and assign ranks, grouping overlapping intervals as ties."""
    rows.sort(key=lambda r: (-r.index if not math.isnan(r.index) else math.inf, r.label))
    # Ranking is idempotent: a row re-ranked after its index changed must not keep the tie
    # partners it had under the old ordering.
    for row in rows:
        row.rank = 0
        row.tied_with = []

    rank = 0
    i = 0
    while i < len(rows):
        if math.isnan(rows[i].index):
            rows[i].rank = 0
            i += 1
            continue
        rank += 1
        group = [rows[i]]
        j = i + 1
        # A model joins the tie group if its interval overlaps ANY member's, which keeps the
        # grouping transitive and avoids the arbitrariness of chaining off the leader alone.
        while tie_on_ci and j < len(rows) and not math.isnan(rows[j].index):
            if any(rows[j].interval.overlaps(g.interval) for g in group):
                group.append(rows[j])
                j += 1
            else:
                break
        for member in group:
            member.rank = rank
            if len(group) > 1:
                member.tied_with = [g.label for g in group if g is not member]
        i = j


def _notes(results: ResultSet, config: RunConfig, rows: list[LeaderboardRow]) -> list[str]:
    """The caveats that must travel with the table."""
    notes = [
        f"The {config.leaderboard.index_name} is a weighted mean of metrics normalised to "
        "0-1 (1 = better) using the range and direction recorded in "
        "`config/benchmarks.yaml`. The weighting is a choice, not a measurement.",
        f"n = {results.metadata.limit} samples per task per model, capped for cost. "
        "This is not a full-benchmark result.",
    ]
    if any(r.tied_with for r in rows):
        notes.append(
            "Models whose 95% intervals overlap share a rank and are marked `=`. "
            "At this sample size that is common, and reporting them as ordered would be a "
            "false claim."
        )
    if mixed := results.mixed_task_versions:
        detail = "; ".join(f"{k}: {sorted(v)}" for k, v in mixed.items())
        notes.append(
            f"**Mixed benchmark versions across cells ({detail}).** Numbers from different "
            "task versions are not directly comparable."
        )
    if partial := [r.label for r in rows if r.partial]:
        notes.append(
            f"Partial coverage for {', '.join(partial)} — the index is computed over the "
            "metrics that produced a value, so it is not strictly comparable with a "
            "complete row."
        )
    blocked = [c for c in results if c.status is CellStatus.BLOCKED]
    if blocked:
        notes.append(
            f"{len(blocked)} cell(s) blocked before running (setup, not model behaviour): "
            f"{', '.join(sorted({c.task_key for c in blocked}))}."
        )
    return notes


def render_markdown(board: Leaderboard) -> str:
    """Render the leaderboard as markdown, for ``results.md`` and the README."""
    header = ["rank", "model", board.index_name] + [
        board.metric_labels[r] for r in board.metric_order
    ]
    lines = [
        f"## {board.index_name}",
        "",
        "Weights: " + ", ".join(f"`{k}` {v:.0%}" for k, v in board.weights.items()),
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in board.rows:
        cells = [row.rank_text, row.label, row.index_text]
        for ref in board.metric_order:
            m = row.metrics.get(ref)
            cells.append(m.format_value() if m else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""] + [f"> {n}" for n in board.notes]
    return "\n".join(lines) + "\n"
