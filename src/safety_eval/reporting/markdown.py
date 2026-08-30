"""``results.md`` — the full record set as a table, one row per (cell, metric).

Columns are exactly the provenance fields of ``docs/SPEC.md`` §4. This is the artefact that
makes the README's claims checkable: every number in a chart or a leaderboard appears here
next to the grader, the seed, the sample counts and the harness versions that produced it.
"""

from __future__ import annotations

import math

from ..config import RunConfig
from ..results import CellStatus, ResultSet


def render_results_markdown(results: ResultSet, config: RunConfig) -> str:
    """Render the full results table with its provenance columns."""
    meta = results.metadata
    counts = results.status_counts()

    lines = [
        "# Results",
        "",
        f"Run `{meta.run_id}` · started {meta.started_utc}"
        + (f" · finished {meta.finished_utc}" if meta.finished_utc else ""),
        "",
        f"- **Harness:** `inspect_ai` {meta.inspect_ai_version}, "
        f"`inspect_evals` {meta.inspect_evals_version}",
        f"- **Provider:** {meta.provider} · **Grader:** `{meta.grader_model}`",
        f"- **Sample cap:** n = {meta.limit} per task per model, capped for cost. "
        "This is not a full-benchmark result.",
        "- **Cells:** " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
        f"- **Cost:** {results.total_tokens:,} tokens, "
        f"{results.total_wall_clock_s / 60:.1f} minutes wall-clock",
    ]
    if meta.notes:
        lines += ["", f"> **{meta.notes}**"]

    lines += [
        "",
        "## Scores",
        "",
        "| task | model | metric | value | 95% CI | scored | unscored | version | grader | "
        "T | seed | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for cell in results:
        if cell.status is not CellStatus.OK:
            lines.append(
                f"| `{cell.task_key}` | {cell.label} | — | — | — | — | — | "
                f"{cell.full_task_version or '—'} | `{_short(cell.grader_model)}` | "
                f"{cell.temperature} | {cell.seed} | **{cell.status.value}**: "
                f"{cell.error_message or ''} |"
            )
            continue
        for m in cell.metrics:
            suffix = "%" if m.unit == "percent" else ""
            value = "nan" if m.is_nan else f"{m.value:.4g}{suffix}"
            ci = "—" if math.isnan(m.ci_low) else f"[{m.ci_low:.3g}, {m.ci_high:.3g}]{suffix}"
            lines.append(
                f"| `{cell.task_key}` | {cell.label} | {m.label} | {value} | {ci} | "
                f"{m.scored_samples} | {m.unscored_samples} | "
                f"{cell.full_task_version or '—'} | `{_short(cell.grader_model)}` | "
                f"{cell.temperature} | {cell.seed} | ok |"
            )

    lines += [
        "",
        "## What is recorded with every score",
        "",
        "A score is a joint property of the model, the harness, the grader, the prompt "
        "sample and the decoding parameters. Publishing the score alone is not "
        "reproducible, so the columns above travel with it.",
        "",
        "`unscored` is the column that costs the most to leave out. Since "
        "`inspect_ai >= 0.3.245`, a grader completion with no parseable `GRADE:` verdict is "
        "`Score.unscored()` rather than a full refusal, so it leaves the metric's "
        "denominator entirely. Grader degradation and genuine compliance now move XSTest's "
        "`refusal_rate` in the *same* direction; `unscored` is the only way to tell them "
        "apart.",
    ]

    if degraded := _degraded(results):
        lines += [
            "",
            "### Cells whose grader under-performed",
            "",
            "The headline number for these cells describes the grader at least as much as "
            "the model:",
            "",
        ]
        lines += [f"- {d}" for d in degraded]

    if mixed := results.mixed_task_versions:
        lines += [
            "",
            "### Mixed benchmark versions",
            "",
            "Numbers from different task versions are not directly comparable:",
            "",
        ]
        lines += [f"- `{k}`: {sorted(v)}" for k, v in mixed.items()]

    withheld = sorted({c.task_key for c in results if not c.log_published})
    if withheld:
        lines += [
            "",
            "### Withheld transcripts",
            "",
            f"Aggregate scores are published for all tasks. Transcripts are published only "
            f"for tasks whose prompts and completions are benign by construction. Logs for "
            f"{', '.join(f'`{w}`' for w in withheld)} are **not** published: they contain "
            "model responses to forbidden prompts. The number is the finding; the "
            "completion that produced it is not something to distribute.",
        ]

    return "\n".join(lines) + "\n"


def _degraded(results: ResultSet, threshold: float = 0.95) -> list[str]:
    out = []
    for cell in results:
        for m in cell.metrics:
            health = m.grader_health
            if not math.isnan(health) and health < threshold:
                out.append(
                    f"`{cell.task_key}` / {cell.label} / {m.label}: grader scored "
                    f"{m.scored_samples} of {m.scored_samples + m.unscored_samples} "
                    f"samples ({health:.0%})"
                )
    return out


def _short(model_id: str | None) -> str:
    return (model_id or "—").rsplit("/", 1)[-1]
