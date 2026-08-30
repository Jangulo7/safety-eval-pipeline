"""Generating the README's results section from a real run.

The ground rule of this repository is that every number in the README comes from a committed
log. Hand-editing the results table is how that rule quietly stops being true, so the section
is generated instead: ``safety-eval report --update-readme`` replaces everything between the
``RESULTS:BEGIN`` and ``RESULTS:END`` markers with a rendering of ``results.json``.

If a run had failures, they appear here too. A README that silently drops the cells that did
not work is not reporting a result, it is advertising one.
"""

from __future__ import annotations

import re
from pathlib import Path

from .leaderboard import Leaderboard
from .results import CellStatus, ResultSet

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


class ReadmeError(RuntimeError):
    """The README could not be updated."""


def render_results_section(
    results: ResultSet,
    board: Leaderboard,
    *,
    gate_passed: bool,
    gate_summary: str,
    chart_rel_path: str | None = None,
) -> str:
    """Render the block that sits between the markers."""
    meta = results.metadata
    if meta.notes:
        raise ReadmeError(
            f"refusing to write results from a run marked {meta.notes!r} into the README"
        )

    counts = results.status_counts()
    lines = [
        f"Run `{meta.run_id}` · {meta.finished_utc or meta.started_utc} · "
        f"`inspect_ai` {meta.inspect_ai_version}, `inspect_evals` {meta.inspect_evals_version}",
        "",
        f"**n = {meta.limit} samples per task per model, capped for cost — not a "
        f"full-benchmark result.** Grader: `{meta.grader_model}`, temperature "
        f"{next((c.temperature for c in results), 0)}, seed {next((c.seed for c in results), 0)}.",
        "",
        f"**Release gate: {'PASS' if gate_passed else 'FAIL'}** — {gate_summary}. "
        "Thresholds are illustrative defaults, not safety claims.",
        "",
    ]

    header = ["rank", "model", board.index_name] + [
        board.metric_labels[r] for r in board.metric_order
    ]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in board.rows:
        cells = [row.rank_text, row.label, row.index_text]
        for ref in board.metric_order:
            m = row.metrics.get(ref)
            cells.append(m.format_value() if m else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "_Weights: " + ", ".join(f"`{k.split(':')[0]}` {v:.0%}"
                                            for k, v in board.weights.items()) + "._"]
    for note in board.notes:
        lines.append(f"_{note}_")

    unscored = [
        f"`{c.task_key}`/{c.label} ({m.unscored_samples}/"
        f"{m.scored_samples + m.unscored_samples})"
        for c in results for m in c.metrics
        if m.unscored_samples and m.grader_health < 0.95
    ]
    if unscored:
        lines += ["", "**Grader degradation** — the headline value for these cells describes "
                      "the grader as much as the model: " + ", ".join(sorted(set(unscored)))
                      + "."]

    problems = [c for c in results if c.status is not CellStatus.OK]
    if problems:
        lines += ["", "### Cells that did not produce a number", "",
                  "A partial matrix with visible gaps is reported rather than a filled-in "
                  "one.", "", "| cell | status | reason |", "|---|---|---|"]
        for c in problems:
            lines.append(f"| `{c.task_key}` / {c.label} | **{c.status.value}** | "
                         f"{(c.error_message or '')[:150]} |")

    if chart_rel_path:
        lines += ["", f"![Over-refusal against under-refusal]({chart_rel_path})", "",
                  "_A single safety score hides a trade-off between over-refusal and "
                  "under-refusal. Top right is the only good corner; movement along the "
                  "diagonal is a trade, not an improvement._"]

    lines += ["", "Full record set with every provenance column: "
                  f"[`results/{meta.run_id}/results.md`](results/{meta.run_id}/results.md) · "
                  f"[gate report](results/{meta.run_id}/gate_report.md) · "
                  f"[leaderboard.html](results/{meta.run_id}/leaderboard.html) · "
                  f"[report.pdf](results/{meta.run_id}/report.pdf)",
              "", f"Cells: {', '.join(f'{v} {k}' for k, v in sorted(counts.items()))} · "
                  f"{results.total_tokens:,} tokens · "
                  f"{results.total_wall_clock_s / 60:.1f} min wall-clock."]
    return "\n".join(lines)


def update_readme(path: Path, section: str) -> Path:
    """Replace the marked block in the README, leaving everything else untouched."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ReadmeError(f"{path} has no {BEGIN} / {END} markers to write between")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    path.write_text(pattern.sub(f"{BEGIN}\n{section}\n{END}", text), encoding="utf-8")
    return path
