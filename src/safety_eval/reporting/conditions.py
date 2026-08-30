"""The run-conditions table: exactly what was measured, under exactly what parameters.

A benchmark score is only comparable across models if every model was measured the same
way, and only reproducible if a reader can see what "the same way" was. Neither is
self-evident from a number, so both are stated explicitly in every artefact.

This module builds the data once and the markdown, HTML and PDF renderers all consume it,
so the three cannot disagree about what the run actually did.

Two checks travel with the table:

* **Divergence** — any parameter that was *not* identical across the models of a task. Read
  from the recorded cells rather than from the config, so a mid-run edit or a resumed run
  cannot slip past.
* **Stratum coverage** — how many of the dataset's own categories each cell actually
  evaluated. These datasets are grouped by category, so a sample cap on an unshuffled
  dataset evaluates one corner of it; the coverage figure is what lets a reader rule that
  out instead of taking it on trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import RunConfig
from ..results import CellStatus, ResultSet


@dataclass
class TaskConditions:
    """The parameters one benchmark was run under, and how well it covered its dataset."""

    task_key: str
    benchmark: str
    task: str
    task_version: str
    task_args: dict[str, Any]
    grader_model: str
    rows: list[tuple[str, str]] = field(default_factory=list)
    strata_covered: int = 0
    strata_total: int = 0
    stratum_counts: dict[str, int] = field(default_factory=dict)
    n_requested: int = 0
    dataset_total: int | None = None
    identical_across_models: bool = True
    divergence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def coverage_text(self) -> str:
        if not self.strata_total:
            return "—"
        pct = ""
        if self.dataset_total and self.n_requested:
            pct = f", {self.n_requested / self.dataset_total:.0%} of {self.dataset_total} samples"
        return f"{self.strata_covered}/{self.strata_total} strata{pct}"

    @property
    def fully_covered(self) -> bool:
        return bool(self.strata_total) and self.strata_covered >= self.strata_total


@dataclass
class Conditions:
    """The whole run's conditions, plus what is not identical."""

    shared: list[tuple[str, str]]
    tasks: list[TaskConditions]
    divergence: dict[str, dict[str, list[str]]]
    under_covered: list[str]

    @property
    def all_identical(self) -> bool:
        return not self.divergence

    @property
    def all_covered(self) -> bool:
        return not self.under_covered


def build(results: ResultSet, config: RunConfig) -> Conditions:
    """Assemble the conditions table from the recorded cells."""
    meta = results.metadata
    divergence = results.condition_divergence()

    first = next((c for c in results if c.ok), None)
    shared = [
        ("Provider", meta.provider),
        ("Grader model", meta.grader_model or "—"),
        ("Samples per cell", results.sample_size_note().replace("**", "")),
        ("Epochs (samples per prompt)", str(getattr(first, "epochs", 1) if first else 1)),
        ("Temperature", str(first.temperature if first else "—")),
        ("Generation seed", str(first.seed if first else "—")),
        ("Dataset-order seed (sample_shuffle)",
         str(first.sample_shuffle) if first and first.sample_shuffle is not None
         else "not set — the cap takes the head of the dataset"),
        ("Max connections", str(first.max_connections if first else "—")),
        ("inspect_ai", meta.inspect_ai_version or "—"),
        ("inspect_evals", meta.inspect_evals_version or "—"),
        ("Pipeline", meta.pipeline_version or "—"),
    ]

    tasks: list[TaskConditions] = []
    for task in config.tasks:
        cells = [c for c in results if c.task_key == task.key]
        ok = [c for c in cells if c.status is CellStatus.OK]
        if not cells:
            continue
        ref = ok[0] if ok else cells[0]
        bench = config.catalog[task.benchmark]
        subset_meta = (bench.subsets or {}).get(task.subset or "", {})

        rows = [
            ("Inspect task", f"`{bench.task}`"),
            ("Task version", str(ref.full_task_version or ref.task_version or "—")),
            ("Task arguments (-T)",
             ", ".join(f"`{k}={v}`" for k, v in sorted(ref.task_args.items())) or "none"),
            ("Grader kwarg", f"`{bench.grader_kwarg}`"),
            ("Grader model", f"`{ref.grader_model}`"),
            ("Samples requested", str(ref.n_requested) if ref.n_requested else "full dataset"),
            ("Epochs", str(ref.epochs)),
            ("Temperature", str(ref.temperature)),
            ("Generation seed", str(ref.seed)),
            ("Dataset-order seed", str(ref.sample_shuffle) if ref.sample_shuffle is not None
             else "not set"),
            ("Dataset", str(bench.dataset.get("source", "—"))),
        ]

        tasks.append(TaskConditions(
            task_key=task.key,
            benchmark=bench.key,
            task=bench.task,
            task_version=str(ref.full_task_version or "—"),
            task_args=dict(ref.task_args),
            grader_model=ref.grader_model or "—",
            rows=rows,
            strata_covered=ref.strata_covered,
            strata_total=ref.strata_total,
            stratum_counts=dict(ref.stratum_counts),
            n_requested=ref.n_requested,
            dataset_total=subset_meta.get("dataset_samples")
            or bench.dataset.get("total_samples"),
            identical_across_models=task.key not in divergence,
            divergence=divergence.get(task.key, {}),
        ))

    under = [
        f"{c.task_key}/{c.label}: {c.strata_covered} of {c.strata_total} strata"
        for c in results.under_covered_cells()
    ]
    return Conditions(shared=shared, tasks=tasks, divergence=divergence, under_covered=under)


PREAMBLE = (
    "Every model ran under the parameters below. They are read back from the recorded "
    "cells, not from the configuration, so a mid-run edit cannot pass unnoticed."
)

COVERAGE_NOTE = (
    "**Stratum coverage** is the share of a dataset's own categories the run evaluated. All "
    "three datasets ship grouped by category, so an unshuffled cap reaches only the first "
    "few: `limit=50` covers 2 of XSTest-safe's 10 prompt types and 1 of StrongREJECT's 6 "
    "harm categories. The dataset-order seed shuffles before the limit applies, which makes "
    "a capped run representative as well as reproducible. The generation seed does not "
    "affect which samples are drawn."
)
