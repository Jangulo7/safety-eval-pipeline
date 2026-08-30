"""The end-to-end pipeline: run -> report -> gate.

Kept separate from the CLI so that the Streamlit dashboard drives exactly the same code
path. Two entry points that produce different artefacts from the same run would defeat the
purpose of having one source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import RunConfig
from .gates import GateReport, evaluate
from .gates import render_markdown as render_gate_markdown
from .leaderboard import Leaderboard
from .leaderboard import build as build_leaderboard
from .leaderboard import render_markdown as render_leaderboard_markdown
from .reporting.charts import ChartSet, render_charts
from .reporting.markdown import render_results_markdown
from .results import CellResult, ResultSet, link_latest, resolve_run_dir
from .runner import Runner


@dataclass
class ReportArtifacts:
    """Everything a report pass produced, and what it could not."""

    run_dir: Path
    results: ResultSet
    board: Leaderboard
    gate_report: GateReport
    charts: ChartSet = field(default_factory=ChartSet)
    files: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return self.gate_report.exit_code


def run_matrix(
    config: RunConfig,
    *,
    run_id: str | None = None,
    progress: Callable[[str, CellResult | None], None] | None = None,
    eval_fn: Callable[..., object] | None = None,
) -> tuple[ResultSet, Path]:
    """Execute the matrix and persist ``results.json``."""
    runner = Runner(config, run_id=run_id, eval_fn=eval_fn)
    results = runner.run(progress=progress)
    run_dir = Path(config.output.results_dir) / runner.run_id
    results.save(run_dir / "results.json")
    link_latest(run_dir)
    return results, run_dir


def report(
    config: RunConfig,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
    results: ResultSet | None = None,
    charts: bool | None = None,
    html: bool | None = None,
    pdf: bool | None = None,
) -> ReportArtifacts:
    """Render every artefact from a run's ``results.json``.

    Nothing here touches a provider or a live log, so a report can be regenerated — and
    audited — long after the run, and a failure in one renderer never loses the others.
    """
    if run_dir is None:
        run_dir = resolve_run_dir(config.output.results_dir, run_id)
    run_dir = Path(run_dir)
    if results is None:
        results = ResultSet.load(run_dir / "results.json")

    board = build_leaderboard(results, config)
    gate_report = evaluate(results, config)
    art = ReportArtifacts(run_dir=run_dir, results=results, board=board,
                          gate_report=gate_report)

    art.files["results_md"] = _write(
        run_dir / "results.md",
        render_results_markdown(results, config) + "\n"
        + render_leaderboard_markdown(board),
    )
    art.files["gate_md"] = _write(
        run_dir / "gate_report.md", render_gate_markdown(gate_report, results, config)
    )

    want_charts = config.output.charts if charts is None else charts
    if want_charts:
        art.charts = render_charts(results, config, board, run_dir / "charts")
        art.skipped.update(art.charts.skipped)

    want_html = config.output.html if html is None else html
    if want_html:
        try:
            from .reporting.html import render_leaderboard_html

            art.files["html"] = _write(
                run_dir / "leaderboard.html",
                render_leaderboard_html(results, config, board, gate_report,
                                        art.charts.paths),
            )
        except Exception as exc:
            art.skipped["html"] = f"{type(exc).__name__}: {exc}"

    want_pdf = config.output.pdf if pdf is None else pdf
    if want_pdf:
        try:
            from .reporting.pdf import build_pdf

            art.files["pdf"] = build_pdf(results, config, board, gate_report,
                                         art.charts.paths, run_dir / "report.pdf")
        except Exception as exc:
            art.skipped["pdf"] = f"{type(exc).__name__}: {exc}"

    return art


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
