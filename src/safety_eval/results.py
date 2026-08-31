"""The result record: the single source of truth every artefact is rendered from.

A score is a joint property of the model, the harness, the grader, the prompt sample and the
decoding parameters. Publishing the score alone is not reproducible, so every field in
``CellResult`` that a reader would need to reproduce or challenge a number is recorded with
it. Charts, the leaderboard, the HTML page and the PDF all render from ``results.json`` and
never from a live log, which means every published artefact can be regenerated — and
audited — without provider access.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


class CellStatus(str, Enum):
    """Outcome of one (model, task) cell.

    ``BLOCKED`` is deliberately distinct from ``ERROR``: a gated dataset or a missing
    credential is a setup problem the operator can fix, while an error is a run problem. A
    report that conflates them sends the reader looking in the wrong place.
    """

    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    """One metric of one cell, with its denominator and interval."""

    address: str
    label: str
    value: float
    ci_low: float
    ci_high: float
    ci_method: str
    scored_samples: int
    unscored_samples: int
    range_low: float
    range_high: float
    direction: str
    unit: str | None = None
    primary: bool = False
    per_stratum: dict[str, list[float]] = field(default_factory=dict)
    """``stratum -> [value, n]``. Reported because an aggregate hides whether a benchmark's
    failure mode is spread evenly or concentrated in one category."""

    normalised: float = math.nan
    """Value mapped to 0-1 good-is-high using the catalog range and direction. This is the
    only form in which metrics from different benchmarks may be combined."""

    @property
    def is_nan(self) -> bool:
        return isinstance(self.value, float) and math.isnan(self.value)

    @property
    def grader_health(self) -> float:
        total = self.scored_samples + self.unscored_samples
        return self.scored_samples / total if total else math.nan


@dataclass
class CellResult:
    """One (model, task) cell: everything needed to reproduce or challenge its numbers."""

    # identity
    run_id: str
    task_key: str
    benchmark: str
    task: str
    task_args: dict[str, Any]
    model_id: str
    family: str
    label: str
    provider: str

    # outcome
    status: CellStatus = CellStatus.OK
    error_message: str | None = None
    metrics: list[MetricResult] = field(default_factory=list)

    # provenance — see docs/SPEC.md §4
    task_version: str | int | None = None
    full_task_version: str | None = None
    grader_model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    """Seeds generation. Distinct from ``sample_shuffle``, which selects the samples."""

    sample_shuffle: int | None = None
    """Seeds the dataset ordering, applied before the limit. Recorded because without it a
    capped run silently evaluates one stratum of a grouped dataset."""

    epochs: int = 1
    max_tokens: int | None = None
    protocol_source: dict[str, str] = field(default_factory=dict)
    """Per generation parameter: ``task`` when it is the benchmark's own published protocol,
    ``pipeline`` when this pipeline had to choose because the task specifies none."""

    applied_generate_config: dict[str, Any] = field(default_factory=dict)
    """The config the harness actually applied, read back from the log rather than assumed.
    A provider that silently discards a parameter is invisible without this."""

    dataset_fingerprint: str | None = None
    """Inspect's own dataset identity hash — evidence that two runs used the same items."""

    dataset_samples_total: int | None = None

    run_commit: str | None = None
    """The commit the run executed at, read from the eval log. Distinct from whatever HEAD
    happens to be when the report is rendered -- reporting the latter is a false provenance
    claim that looks precise."""

    run_commit_dirty: bool | None = None
    """Whether the working tree carried uncommitted changes during the run. A commit alone
    does not reproduce a dirty tree, so the flag travels with it."""

    system_prompt: str | None = None
    """The system message actually sent, read back from the log's first sample. ``None``
    means the log shows no system message; it does not mean unknown."""

    eval_config: dict[str, Any] = field(default_factory=dict)
    """The run configuration the harness recorded, including epochs and the dataset-order
    seed. Measured, not restated from our own config."""

    serving: dict[str, Any] = field(default_factory=dict)
    """What the serving stack reported about itself at run time -- engine version, dtype,
    quantization, context length. Empty when it could not be queried."""
    max_connections: int | None = None
    n_requested: int = 0
    n_completed: int = 0

    strata_covered: int = 0
    strata_total: int = 0
    stratum_counts: dict[str, int] = field(default_factory=dict)
    """Measured coverage of the dataset's own categories. A cap that touches one stratum is
    not a run of the benchmark, whatever the headline metric says."""
    inspect_ai_version: str | None = None
    inspect_evals_version: str | None = None

    # cost and timing
    wall_clock_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # artefacts
    log_path: str | None = None
    log_published: bool = False
    """False for any task whose catalog entry sets ``publish_logs: false``. StrongREJECT
    transcripts contain harmful completions and are never written to a published artefact —
    see docs/SPEC.md §9."""

    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    attempts: int = 1

    def metric(self, address: str) -> MetricResult | None:
        """Look up a metric by address, or by its short key when unambiguous."""
        for m in self.metrics:
            if m.address == address:
                return m
        matches = [m for m in self.metrics if m.address.rsplit("/", 1)[-1] == address]
        return matches[0] if len(matches) == 1 else None

    @property
    def ok(self) -> bool:
        return self.status is CellStatus.OK

    @property
    def cell_id(self) -> str:
        return f"{self.task_key}::{self.model_id}"

    @property
    def conditions(self) -> dict[str, Any]:
        """The exact parameters this cell was run under, for the report's conditions table."""
        return {
            "limit": self.n_requested,
            "epochs": self.epochs,
            "temperature": self.temperature,
            "seed": self.seed,
            "sample_shuffle": self.sample_shuffle,
            "max_tokens": self.max_tokens,
            "max_connections": self.max_connections,
            "grader_model": self.grader_model,
            "task_args": dict(self.task_args),
            "task_version": self.full_task_version,
            "inspect_ai": self.inspect_ai_version,
            "inspect_evals": self.inspect_evals_version,
        }

    @property
    def stratum_coverage(self) -> float:
        """Fraction of the dataset's strata this cell actually evaluated."""
        return self.strata_covered / self.strata_total if self.strata_total else math.nan


@dataclass
class RunMetadata:
    """Run-level provenance shared by every cell."""

    run_id: str
    started_utc: str
    finished_utc: str | None = None
    config_path: str | None = None
    catalog_path: str | None = None
    provider: str = "openrouter"
    grader_model: str | None = None
    limit: int = 0
    inspect_ai_version: str | None = None
    inspect_evals_version: str | None = None
    pipeline_version: str | None = None
    command: str | None = None
    host: dict[str, Any] = field(default_factory=dict)
    """Hardware the run executed on, queried at run time. Empty when not captured -- which
    the report states, rather than printing a placeholder."""

    notes: str = ""


class ResultSet:
    """A run's cells plus its metadata, with JSON round-tripping."""

    def __init__(self, metadata: RunMetadata, cells: list[CellResult] | None = None) -> None:
        self.metadata = metadata
        self.cells: list[CellResult] = cells or []

    # ------------------------------------------------------------------ mutation

    def add(self, cell: CellResult) -> None:
        self.cells.append(cell)

    # ------------------------------------------------------------------ querying

    def __iter__(self) -> Iterator[CellResult]:
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def ok_cells(self) -> list[CellResult]:
        return [c for c in self.cells if c.ok]

    @property
    def models(self) -> list[str]:
        """Model ids in first-seen order (not sorted: config order is meaningful)."""
        seen: dict[str, None] = {}
        for c in self.cells:
            seen.setdefault(c.model_id, None)
        return list(seen)

    @property
    def task_keys(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.cells:
            seen.setdefault(c.task_key, None)
        return list(seen)

    def label_for(self, model_id: str) -> str:
        for c in self.cells:
            if c.model_id == model_id:
                return c.label
        return model_id

    def get(self, task_key: str, model_id: str) -> CellResult | None:
        for c in self.cells:
            if c.task_key == task_key and c.model_id == model_id:
                return c
        return None

    def value(self, task_key: str, model_id: str, metric_address: str) -> float:
        """A single metric value, or ``nan`` when the cell failed or the metric is absent.

        ``nan`` rather than ``0.0`` on purpose: a missing number must never be plotted or
        ranked as a good one.
        """
        cell = self.get(task_key, model_id)
        if cell is None or not cell.ok:
            return math.nan
        m = cell.metric(metric_address)
        return m.value if m else math.nan

    def condition_divergence(self) -> dict[str, dict[str, list[str]]]:
        """Per task, any run condition that was NOT identical across models.

        A cross-model comparison is only a comparison if the conditions were the same. This
        is checked from the recorded cells rather than assumed from the config, so a mid-run
        config edit or a resumed run cannot slip through.
        """
        out: dict[str, dict[str, list[str]]] = {}
        for task_key in self.task_keys:
            cells = [c for c in self.cells if c.task_key == task_key and c.ok]
            if len(cells) < 2:
                continue
            diverged: dict[str, list[str]] = {}
            for field_name in ("limit", "epochs", "temperature", "seed", "sample_shuffle",
                               "grader_model", "task_args", "task_version"):
                values = {repr(c.conditions[field_name]) for c in cells}
                if len(values) > 1:
                    diverged[field_name] = sorted(values)
            if diverged:
                out[task_key] = diverged
        return out

    def under_covered_cells(self, threshold: float = 1.0) -> list[CellResult]:
        """Cells that evaluated fewer than ``threshold`` of their dataset's strata."""
        return [
            c for c in self.cells
            if c.ok and c.strata_total and c.stratum_coverage < threshold
        ]

    @property
    def mixed_task_versions(self) -> dict[str, set[str]]:
        """Task keys whose cells did not all run the same benchmark version.

        A comparison across differing task versions is not a comparison; the leaderboard
        footnotes this rather than silently ranking incomparable numbers.
        """
        by_task: dict[str, set[str]] = {}
        for c in self.cells:
            if c.ok and c.full_task_version:
                by_task.setdefault(c.task_key, set()).add(str(c.full_task_version))
        return {k: v for k, v in by_task.items() if len(v) > 1}

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.cells)

    @property
    def total_wall_clock_s(self) -> float:
        return sum(c.wall_clock_s for c in self.cells)

    def sample_sizes(self) -> dict[str, tuple[int, bool]]:
        """Per task: ``(n, was_capped)``, read from the cells rather than the config.

        A run-level sample cap stopped being meaningful once tasks began capping themselves
        independently — XSTest runs in full at 250 while sycophancy is capped at 250 of
        4,882 — so the size of the evidence is reported per benchmark, and whether it was
        capped is reported with it.
        """
        out: dict[str, tuple[int, bool]] = {}
        for cell in self.cells:
            if not cell.ok:
                continue
            capped = bool(cell.dataset_samples_total
                          and cell.n_completed < cell.dataset_samples_total)
            out[cell.task_key] = (cell.n_completed, capped)
        return out

    def sample_size_note(self) -> str:
        """One sentence stating the evidence behind every number, honestly per task."""
        sizes = self.sample_sizes()
        if not sizes:
            return "No completed cells."
        full = [k for k, (_, capped) in sizes.items() if not capped]
        capped = [k for k, (_, capped) in sizes.items() if capped]
        parts = [f"`{k}` n = {n}" for k, (n, _) in sorted(sizes.items())]
        note = "Samples per model: " + ", ".join(parts) + "."
        if full:
            note += (f" {', '.join('`' + f + '`' for f in full)} ran the **full dataset**, "
                     "so those numbers are the benchmark's score rather than a sample of it.")
        if capped:
            note += (f" {', '.join('`' + c + '`' for c in capped)} was capped for cost, with "
                     "a seeded dataset-order shuffle so the subset covers every stratum.")
        return note

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.cells:
            counts[c.status.value] = counts.get(c.status.value, 0) + 1
        return counts

    # ------------------------------------------------------------------ IO

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": asdict(self.metadata),
            "cells": [_cell_to_dict(c) for c in self.cells],
        }

    def save(self, path: str | Path) -> Path:
        """Write ``results.json``. ``nan`` is serialised as ``null`` with a sibling flag."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ResultSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema_version {version}, this build reads {SCHEMA_VERSION}"
            )
        meta = RunMetadata(**data["metadata"])
        cells = [_cell_from_dict(c) for c in data["cells"]]
        return cls(meta, cells)

    def to_rows(self) -> list[dict[str, Any]]:
        """Flatten to one row per (cell, metric) for tables and dataframes."""
        rows: list[dict[str, Any]] = []
        for c in self.cells:
            base = {
                "task": c.task_key,
                "benchmark": c.benchmark,
                "model": c.label,
                "model_id": c.model_id,
                "family": c.family,
                "status": c.status.value,
                "task_version": c.full_task_version or c.task_version,
                "grader_model": c.grader_model,
                "temperature": c.temperature,
                "seed": c.seed,
                "n_requested": c.n_requested,
                "n_completed": c.n_completed,
                "epochs": c.epochs,
                "sample_shuffle": c.sample_shuffle,
                "strata": (f"{c.strata_covered}/{c.strata_total}"
                           if c.strata_total else "—"),
                "wall_clock_s": round(c.wall_clock_s, 1),
                "total_tokens": c.total_tokens,
                "inspect_ai": c.inspect_ai_version,
                "inspect_evals": c.inspect_evals_version,
                "log_published": c.log_published,
            }
            if not c.metrics:
                rows.append({**base, "metric": None, "value": None, "ci": None,
                             "scored": None, "unscored": None,
                             "error": c.error_message})
                continue
            for m in c.metrics:
                rows.append({
                    **base,
                    "metric": m.label,
                    "metric_address": m.address,
                    "value": None if m.is_nan else m.value,
                    "unit": m.unit,
                    "ci": None if math.isnan(m.ci_low) else f"[{m.ci_low:.3g}, {m.ci_high:.3g}]",
                    "ci_low": None if math.isnan(m.ci_low) else m.ci_low,
                    "ci_high": None if math.isnan(m.ci_high) else m.ci_high,
                    "ci_method": m.ci_method,
                    "scored": m.scored_samples,
                    "unscored": m.unscored_samples,
                    "direction": m.direction,
                    "error": c.error_message,
                })
        return rows


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def _cell_to_dict(cell: CellResult) -> dict[str, Any]:
    d = asdict(cell)
    d["status"] = cell.status.value
    # json.dumps writes bare NaN, which is invalid JSON and breaks non-Python readers.
    for m in d["metrics"]:
        for key in ("value", "ci_low", "ci_high", "normalised"):
            if isinstance(m[key], float) and math.isnan(m[key]):
                m[key] = None
    return d


def _cell_from_dict(d: dict[str, Any]) -> CellResult:
    d = dict(d)
    metrics = []
    for m in d.pop("metrics", []):
        m = dict(m)
        for key in ("value", "ci_low", "ci_high", "normalised"):
            if m.get(key) is None:
                m[key] = math.nan
        metrics.append(MetricResult(**m))
    d["status"] = CellStatus(d.get("status", "ok"))
    return CellResult(metrics=metrics, **d)


def new_run_id(prefix: str = "run") -> str:
    """A sortable run id: ``run-20260830-181500``."""
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def resolve_run_dir(results_dir: str | Path, run_id: str | None = None) -> Path:
    """Locate a run directory, defaulting to the most recent one.

    ``latest`` is a symlink where the filesystem allows it and a plain directory copy where
    it does not, so this falls back to lexical ordering of run ids.
    """
    results_dir = Path(results_dir)
    if run_id:
        path = results_dir / run_id
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} under {results_dir}")
        return path
    latest = results_dir / "latest"
    if latest.exists():
        return latest.resolve()
    runs = sorted(p for p in results_dir.glob("run-*") if (p / "results.json").exists())
    if not runs:
        raise FileNotFoundError(
            f"no runs found under {results_dir}. Run `safety-eval run` first."
        )
    return runs[-1]


def link_latest(run_dir: Path) -> None:
    """Point ``results/latest`` at ``run_dir``, tolerating filesystems without symlinks."""
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink():
                latest.unlink()
            else:
                return  # a real directory is there; do not clobber it
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass  # Windows without developer mode; resolve_run_dir falls back to lexical order
