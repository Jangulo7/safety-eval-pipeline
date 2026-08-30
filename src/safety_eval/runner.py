"""Execution of the (model x task) matrix against the Inspect harness.

Design commitments, all of which exist because a long provider-backed run is expensive to
repeat:

**Per-cell isolation.** One model being rate-limited, unavailable or gated must not lose the
other eleven cells. Every cell is wrapped; a failure is recorded with its message and the run
continues. A partial matrix with visible gaps is honest. A crashed run is not.

**Blocked is not error.** A gated dataset or a missing credential is classified as
``BLOCKED`` and detected *before* any generation happens, so the operator can fix it without
having paid for the attempt.

**Retry once, then record.** Transient provider errors are worth one retry. More than that
turns a broken run into an expensive broken run.

**Dry run spends nothing.** ``--dry-run`` resolves the whole matrix and prints what it would
do without constructing a task or opening a connection.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from .catalog import BenchmarkSpec, Catalog
from .config import ModelSpec, RunConfig, TaskSpec
from .metrics import read_all
from .results import (
    CellResult,
    CellStatus,
    MetricResult,
    ResultSet,
    RunMetadata,
    link_latest,
    new_run_id,
)

log = logging.getLogger(__name__)

# Substrings that identify a setup problem rather than a run problem. Matching on message
# text is unlovely, but the alternative — importing exception classes from `datasets`,
# `huggingface_hub` and every provider SDK — couples this module to all of them.
_BLOCKED_SIGNATURES = (
    "gated dataset",
    "gated repo",
    "is a gated",
    "ask for access",
    "401 client error",
    "403 client error",
    "authentication",
    "api key",
    "unauthorized",
    "not found on the hub",
    "datasetnotfounderror",
)

_TRANSIENT_SIGNATURES = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "503",
    "502",
    "overloaded",
)


@dataclass
class CellPlan:
    """One resolved cell, before it is run. This is what ``--dry-run`` prints."""

    model: ModelSpec
    task: TaskSpec
    benchmark: BenchmarkSpec
    task_args: dict[str, Any]
    limit: int
    log_dir: Path
    publish_logs: bool

    @property
    def cell_id(self) -> str:
        return f"{self.task.key}::{self.model.id}"


class Runner:
    """Runs the matrix and produces a :class:`ResultSet`."""

    def __init__(
        self,
        config: RunConfig,
        catalog: Catalog | None = None,
        *,
        run_id: str | None = None,
        eval_fn: Callable[..., Any] | None = None,
    ) -> None:
        """
        Args:
            config: the validated run configuration.
            catalog: the benchmark catalog (defaults to the one the config validated against).
            run_id: override the generated run id, for deterministic tests.
            eval_fn: injection point for ``inspect_ai.eval``. Tests substitute a stub so the
                whole matrix, its error handling and its retry logic are exercised offline.
        """
        self.config = config
        self.catalog = catalog or config.catalog
        self.run_id = run_id or new_run_id()
        self._eval_fn = eval_fn
        self.versions = harness_versions()

    # ------------------------------------------------------------------ planning

    def plan(self) -> list[CellPlan]:
        """Resolve every cell without touching a provider or a dataset."""
        d = self.config.defaults
        plans: list[CellPlan] = []
        for task, model in [(t, m) for t in self.config.tasks for m in self.config.models]:
            bench = self.catalog[task.benchmark]
            plans.append(
                CellPlan(
                    model=model,
                    task=task,
                    benchmark=bench,
                    task_args=dict(task.args),
                    limit=d.limit,
                    log_dir=self.config.output.log_dir / task.slug / model.slug,
                    publish_logs=bench.publish_logs,
                )
            )
        return plans

    def describe_plan(self) -> str:
        """A human-readable dry-run summary, including what will *not* be published."""
        plans = self.plan()
        d = self.config.defaults
        lines = [
            f"run_id            {self.run_id}",
            f"provider          {self.config.provider}",
            f"grader            {self.config.grader_model}",
            f"harness           inspect_ai {self.versions['inspect_ai']}, "
            f"inspect_evals {self.versions['inspect_evals']}",
            f"cells             {len(plans)} ({len(self.config.models)} models x "
            f"{len(self.config.tasks)} tasks)",
            f"samples requested {len(plans) * d.limit} "
            f"(limit={d.limit}, temperature={d.temperature}, seed={d.seed})",
            "",
            f"{'cell':<52} {'n':>4}  logs",
        ]
        for p in plans:
            published = "published" if p.publish_logs else "WITHHELD (harmful completions)"
            lines.append(f"{p.cell_id:<52} {p.limit:>4}  {published}")
        gated = sorted({p.benchmark.key for p in plans if p.benchmark.gated})
        if gated:
            lines += ["", f"gated datasets: {', '.join(gated)} — run `safety-eval doctor` first"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ execution

    def run(
        self,
        progress: Callable[[str, CellResult | None], None] | None = None,
    ) -> ResultSet:
        """Execute every cell, isolating failures. Returns the completed result set.

        Args:
            progress: called as ``progress(cell_id, result_or_None)`` — with ``None`` when a
                cell starts and with the result when it finishes. The Streamlit UI and the
                CLI both drive their output from this.
        """
        from datetime import datetime

        meta = RunMetadata(
            run_id=self.run_id,
            started_utc=datetime.now(UTC).isoformat(),
            config_path=str(self.config.source) if self.config.source else None,
            provider=self.config.provider,
            grader_model=self.config.grader_model,
            limit=self.config.defaults.limit,
            inspect_ai_version=self.versions["inspect_ai"],
            inspect_evals_version=self.versions["inspect_evals"],
            pipeline_version=self.versions["safety_eval"],
        )
        results = ResultSet(meta)

        for plan in self.plan():
            if progress:
                progress(plan.cell_id, None)
            cell = self._run_cell(plan)
            results.add(cell)
            if progress:
                progress(plan.cell_id, cell)

        meta.finished_utc = datetime.now(UTC).isoformat()
        return results

    def _run_cell(self, plan: CellPlan) -> CellResult:
        """Run one cell with retry, classifying any failure."""
        cell = self._blank_cell(plan)
        attempts = 0
        max_attempts = 1 + max(0, self.config.defaults.max_retries)
        started = time.monotonic()

        while attempts < max_attempts:
            attempts += 1
            try:
                eval_log = self._invoke(plan)
                cell = self._from_log(plan, eval_log)
                cell.attempts = attempts
                cell.wall_clock_s = time.monotonic() - started
                return cell
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                log.warning("cell %s attempt %d failed: %s", plan.cell_id, attempts, message)
                log.debug("%s", traceback.format_exc())
                status = classify_failure(message)
                if status is CellStatus.BLOCKED or attempts >= max_attempts:
                    cell.status = status
                    cell.error_message = _truncate(message)
                    cell.attempts = attempts
                    cell.wall_clock_s = time.monotonic() - started
                    return cell
                time.sleep(min(10.0, 2.0 ** attempts))

        return cell

    def _invoke(self, plan: CellPlan) -> Any:
        """Call the Inspect harness for one cell."""
        eval_fn = self._eval_fn
        if eval_fn is None:
            from inspect_ai import eval as eval_fn

        d = self.config.defaults
        plan.log_dir.mkdir(parents=True, exist_ok=True)

        logs = eval_fn(
            tasks=plan.benchmark.task,
            task_args=plan.task_args,
            model=plan.model.id,
            limit=plan.limit,
            max_connections=d.max_connections,
            temperature=d.temperature,
            seed=d.seed,
            log_dir=str(plan.log_dir),
            fail_on_error=False,     # a bad sample is data, not a reason to lose the cell
            display="plain",
        )
        if not logs:
            raise RuntimeError("inspect_ai.eval returned no logs")
        return logs[0]

    # ------------------------------------------------------------------ conversion

    def _blank_cell(self, plan: CellPlan) -> CellResult:
        d = self.config.defaults
        return CellResult(
            run_id=self.run_id,
            task_key=plan.task.key,
            benchmark=plan.benchmark.key,
            task=plan.benchmark.task,
            task_args=plan.task_args,
            model_id=plan.model.id,
            family=plan.model.family,
            label=plan.model.label,
            provider=plan.model.provider,
            grader_model=self.config.grader_model,
            temperature=d.temperature,
            seed=d.seed,
            max_connections=d.max_connections,
            n_requested=plan.limit,
            inspect_ai_version=self.versions["inspect_ai"],
            inspect_evals_version=self.versions["inspect_evals"],
            log_published=plan.publish_logs,
        )

    def _from_log(self, plan: CellPlan, eval_log: Any) -> CellResult:
        """Turn a completed ``EvalLog`` into a :class:`CellResult`."""
        cell = self._blank_cell(plan)
        spec = getattr(eval_log, "eval", None)

        if getattr(eval_log, "status", None) == "error":
            cell.status = CellStatus.ERROR
            cell.error_message = _truncate(str(getattr(eval_log, "error", "eval failed")))
            return cell

        if spec is not None:
            cell.task_version = spec.task_version
            cell.full_task_version = (spec.metadata or {}).get("full_task_version")
            cell.log_path = str(getattr(eval_log, "location", "") or "")

        results = getattr(eval_log, "results", None)
        if results is not None:
            cell.n_completed = int(getattr(results, "completed_samples", 0) or 0)

        stats = getattr(eval_log, "stats", None)
        if stats is not None:
            for usage in (stats.model_usage or {}).values():
                cell.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                cell.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                cell.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

        subset = plan.task.subset or plan.task_args.get("subset")
        readings = read_all(eval_log, plan.benchmark.metrics.values(), seed=self.config.defaults.seed)
        for address, reading in readings.items():
            m = plan.benchmark.metrics[address]
            cell.metrics.append(
                MetricResult(
                    address=address,
                    label=m.label,
                    value=reading.value,
                    ci_low=reading.interval.low,
                    ci_high=reading.interval.high,
                    ci_method=reading.interval.method,
                    scored_samples=reading.scored_samples,
                    unscored_samples=reading.unscored_samples,
                    range_low=m.range[0],
                    range_high=m.range[1],
                    direction=m.direction_for(subset).value,
                    unit=m.unit,
                    primary=m.primary,
                    normalised=m.normalise(reading.value, subset),
                )
            )

        if not cell.metrics:
            cell.status = CellStatus.ERROR
            cell.error_message = (
                "run completed but no catalog metric could be read from the log — "
                "the harness version may have renamed a score. Re-run "
                "`safety-eval doctor --metrics`."
            )
        return cell


def classify_failure(message: str) -> CellStatus:
    """Distinguish a setup problem (``BLOCKED``) from a run problem (``ERROR``).

    Blocked cells are worth reporting differently: they tell the operator to go and grant
    dataset access or fix a key, and retrying them costs money for a guaranteed failure.
    """
    lowered = message.lower()
    if any(sig in lowered for sig in _BLOCKED_SIGNATURES):
        return CellStatus.BLOCKED
    return CellStatus.ERROR


def is_transient(message: str) -> bool:
    """Whether a failure is worth the one retry."""
    lowered = message.lower()
    return any(sig in lowered for sig in _TRANSIENT_SIGNATURES)


def harness_versions() -> dict[str, str]:
    """Resolved versions of everything a published number depends on.

    Recorded with every result because a score is a property of the harness as much as of
    the model — an ``inspect_evals`` upgrade that changes a scorer changes the number.
    """
    out: dict[str, str] = {}
    for name, dist in (("inspect_ai", "inspect_ai"), ("inspect_evals", "inspect_evals"),
                       ("safety_eval", "llm-nightly-eval")):
        try:
            out[name] = pkg_version(dist)
        except PackageNotFoundError:
            out[name] = "unknown"
    if out.get("safety_eval") == "unknown":
        from . import __version__
        out["safety_eval"] = __version__
    return out


def execute(
    config: RunConfig,
    *,
    run_id: str | None = None,
    progress: Callable[[str, CellResult | None], None] | None = None,
    eval_fn: Callable[..., Any] | None = None,
) -> tuple[ResultSet, Path]:
    """Run the matrix and persist ``results.json``. Returns the results and the run directory."""
    runner = Runner(config, run_id=run_id, eval_fn=eval_fn)
    results = runner.run(progress=progress)
    run_dir = config.output.results_dir / runner.run_id
    results.save(run_dir / "results.json")
    link_latest(run_dir)
    return results, run_dir


def _truncate(message: str, limit: int = 500) -> str:
    message = " ".join(message.split())
    return message if len(message) <= limit else message[: limit - 1] + "…"
