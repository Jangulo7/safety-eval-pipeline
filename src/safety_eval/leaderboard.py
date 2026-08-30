"""Ordering models by how many hard requirements they violate.

**Models are ranked by gate failures, not by a composite score.** The composite index is
computed and displayed, but only as a diagnostic; it does not determine the order. That is a
deliberate reversal of the obvious design, for reasons that are structural rather than
matters of taste:

* **A weighted mean is compensatory and safety is not.** A model that is flawless everywhere
  except producing maximally specific harmful content on every forbidden prompt still scores
  0.700 of 1.0 under the configured weights, because the other metrics pay for it. Being
  excellent at one thing should not buy permission to be catastrophic at another.
* **Two of the metrics are the same measurement inverted, so they cancel.** XSTest-safe and
  XSTest-unsafe are one metric over two subsets in opposite directions. A model that refuses
  everything, a model that refuses nothing, and a model that coin-flips all contribute
  identically (0.250). Those are precisely the failure modes this pipeline exists to
  distinguish.
* **The metrics are not commensurable.** A proportion of prompts, a severity-weighted mean on
  0-5, and a rate conditioned on a per-model subset are different kinds of quantity. Mapping
  them linearly onto 0-1 makes them look comparable without making them so.

Counting violated thresholds has none of those problems: it is non-compensatory by
construction, it needs no weights, and a reader knows exactly what "two failures" means.

Constraints that still apply:

1. **Models violating the same number of requirements share a rank.** There is no basis in
   this data for ordering them, and inventing one is what the index did.
2. **A model with a missing cell is not silently advantaged.** Coverage is reported and the
   row is marked partial.
3. **The index keeps its interval and its caveats** wherever it is shown, so nobody mistakes
   a diagnostic for a measurement.
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
    gate_failures: list[str] = field(default_factory=list)
    """Gates this model breached. The index is computed from aggregate metrics only, so it
    cannot see a category-level failure; a model can therefore rank top while failing a
    gate. Carrying the gate outcome on the row makes that impossible to miss."""

    @property
    def gate_text(self) -> str:
        if not self.gate_failures:
            return "pass"
        return f"FAIL ({len(self.gate_failures)})"

    @property
    def partial(self) -> bool:
        return self.coverage < 1.0

    @property
    def index_text(self) -> str:
        return "—" if math.isnan(self.index) else f"{self.index:.3f}"

    @property
    def rank_text(self) -> str:
        """Rank, with an equals sign when shared. `=1` reads as 'joint first'."""
        if self.rank == 0:
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


def build(results: ResultSet, config: RunConfig, gate_report=None) -> Leaderboard:
    """Compute the composite index and rank the models.

    Args:
        gate_report: an optional evaluated :class:`~safety_eval.gates.GateReport`. When
            given, each row carries its gate outcome — necessary because the index is built
            from aggregate metrics and is blind to the per-stratum failures the gate
            catches, so the top-ranked model may be one that fails a gate.
    """
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

    failures: dict[str, list[str]] = {}
    if gate_report is not None:
        for failure in gate_report.failures:
            failures.setdefault(failure.model_id, []).append(failure.gate_id)
        for row in rows:
            row.gate_failures = failures.get(row.model_id, [])

    _rank(rows, gated=gate_report is not None)

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


def _rank(rows: list[LeaderboardRow], *, gated: bool = True) -> None:
    """Order by the number of hard requirements violated; equal counts share a rank.

    Non-compensatory by construction: nothing a model does well can reduce its count of
    violated thresholds. Models with the same count are *not* separated by the composite
    index — doing that would smuggle the compensatory ordering back in through the tiebreak,
    which is the whole thing this replaces.
    """
    rows.sort(key=lambda r: (len(r.gate_failures), r.label))
    for row in rows:
        row.rank = 0
        row.tied_with = []

    if not gated:
        return

    rank = 0
    i = 0
    while i < len(rows):
        count = len(rows[i].gate_failures)
        group = [r for r in rows if len(r.gate_failures) == count]
        rank += 1
        for member in group:
            member.rank = rank
            if len(group) > 1:
                member.tied_with = [g.label for g in group if g is not member]
        i += len(group)


def _notes(results: ResultSet, config: RunConfig, rows: list[LeaderboardRow]) -> list[str]:
    """The caveats that must travel with the table."""
    notes = [
        results.sample_size_note(),
        "Rank counts violated thresholds, including per-stratum bounds. It is not a "
        "quality score, and rank 1 is not a pass.",
    ]
    if any(r.tied_with for r in rows):
        notes.append(
            "Models marked `=` violate the same number of thresholds. Separating them "
            "further would require a composite score, which this report does not rank on."
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
    compressed = _compressed_metrics(results, config)
    if compressed:
        notes.append(
            "The index normalises over each metric's **declared** range, and the models "
            "occupy only part of it: " + "; ".join(compressed) + ". Narrowing a declared "
            "range changes the index and can reorder models. Per-benchmark scores and gates "
            "use native units and are unaffected."
        )

    blocked = [c for c in results if c.status is CellStatus.BLOCKED]
    if blocked:
        notes.append(
            f"{len(blocked)} cell(s) blocked before running (setup, not model behaviour): "
            f"{', '.join(sorted({c.task_key for c in blocked}))}."
        )
    return notes


def _compressed_metrics(results: ResultSet, config: RunConfig) -> list[str]:
    """Weighted metrics whose observed spread is small against their declared range.

    Reported as a fraction of the range rather than as a ratio between models. A ratio is
    the wrong summary here: two models at 0.121 and 1.468 on a 0-5 scale differ by 12x, but
    that number is large only because the smaller one is near zero, and it says nothing
    about how far apart they are on the scale the index normalises over. The fraction does.
    """
    out = []
    for ref in config.leaderboard.weights:
        task_key, address = ref.split(":", 1)
        values = []
        for model_id in results.models:
            cell = results.get(task_key, model_id)
            m = cell.metric(address) if cell and cell.ok else None
            if m and not m.is_nan:
                values.append(m)
        if len(values) < 2:
            continue
        lo, hi = values[0].range_low, values[0].range_high
        obs = [m.value for m in values]
        span = (max(obs) - min(obs)) / (hi - lo) if hi > lo else 0.0
        if span < 0.35:
            worst = max(obs) if values[0].direction == "lower_better" else min(obs)
            out.append(f"`{task_key}` uses {span:.0%} of its {lo:g}-{hi:g} range "
                       f"(worst model {worst:.3g})")
    return out


def render_markdown(board: Leaderboard) -> str:
    """Render the leaderboard as markdown, for ``results.md`` and the README."""
    header = ["rank", "model", "gate failures"] + [
        board.metric_labels[r] for r in board.metric_order
    ] + [f"{board.index_name} (diagnostic)"]
    lines = [
        "## Ranking",
        "",
        "**Models rank by the number of gate thresholds they violate, not by a composite "
        "score.** Counting violations is non-compensatory: strong performance on one axis "
        "cannot offset a failure on another. Equal counts share a rank.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in board.rows:
        cells = [row.rank_text, row.label,
                 ", ".join(f"`{g}`" for g in row.gate_failures) or "none"]
        for ref in board.metric_order:
            m = row.metrics.get(ref)
            cells.append(m.format_value() if m else "—")
        cells.append(row.index_text)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""] + [f"> {n}" for n in board.notes]
    lines += [
        "",
        f"### {board.index_name} — diagnostic only, not a ranking",
        "",
        "Weights: " + ", ".join(f"`{k}` {v:.0%}" for k, v in board.weights.items()) + ".",
        "",
        "A normalised cross-benchmark summary makes a useful smell test. It does not make "
        "a sound ranking. Three structural problems, each measurable in this run:",
        "",
        "1. **It is compensatory; safety is not.** A model flawless everywhere except "
        "producing maximally specific harmful content on every forbidden prompt still "
        "scores 0.700 of 1.0 under these weights.",
        "2. **Two inputs are the same metric inverted, so they cancel.** Refusing "
        "everything, refusing nothing, and coin-flipping all contribute 0.250. "
        "Distinguishing those is why both XSTest subsets run.",
        "3. **The inputs are not commensurable.** A proportion of prompts, a "
        "severity-weighted mean on 0-5, and a rate conditioned on a per-model subset are "
        "different quantities. A linear map to 0-1 does not make them comparable.",
    ]
    return "\n".join(lines) + "\n"
