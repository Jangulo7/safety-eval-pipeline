"""``results.md`` — the full record set as a table, one row per (cell, metric).

Columns are exactly the provenance fields of ``docs/SPEC.md`` §4. This is the artefact that
makes the README's claims checkable: every number in a chart or a leaderboard appears here
next to the grader, the seed, the sample counts and the harness versions that produced it.
"""

from __future__ import annotations

import math

from ..config import RunConfig
from ..disclosure import (
    DISCLOSURE_SCHEMA_CITATION,
    DISCLOSURE_SCHEMA_URL,
    contamination_disclosure,
    parameter_register,
)
from ..results import CellStatus, ResultSet
from .conditions import COVERAGE_NOTE, PREAMBLE
from .conditions import build as build_conditions


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
        f"- **Sample size:** {results.sample_size_note()}",
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

    lines += _stratum_section(results, config)
    lines += _conditions_section(results, config)
    lines += _disclosure_section(results, config)
    lines += _register_section(results, config)

    lines += [
        "",
        "## What is recorded with every score",
        "",
        "A score is a joint property of the model, the harness, the grader, the prompt "
        "sample and the decoding parameters. The score alone is not reproducible, so the "
        "columns above travel with it.",
        "",
        "`unscored` costs the most to omit. Since `inspect_ai >= 0.3.245`, an unparseable "
        "grader verdict yields `Score.unscored()` rather than a full refusal, so the sample "
        "leaves the denominator. Grader degradation and genuine compliance now move "
        "`refusal_rate` in the same direction. Only `unscored` separates them.",
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


def _stratum_section(results: ResultSet, config: RunConfig) -> list[str]:
    """Per-category scores. An aggregate hides where a benchmark's failure mode lives."""
    lines: list[str] = []
    for task in config.tasks:
        rows = []
        for model_id in results.models:
            cell = results.get(task.key, model_id)
            if cell is None or not cell.ok:
                continue
            for m in cell.metrics:
                if m.per_stratum:
                    rows.append((cell.label, m, m.per_stratum))
        if not rows:
            continue
        if not lines:
            lines += ["", "## Scores by stratum", "",
                      "Each benchmark broken down by its own categories. An aggregate "
                      "refusal rate cannot say whether refusals spread evenly or "
                      "concentrated in one prompt type. For a benchmark built to locate "
                      "over-refusal, that distinction is the finding.", ""]
        strata = sorted({s for _, _, ps in rows for s in ps})
        suffix = "%" if rows[0][1].unit == "percent" else ""
        lines += [f"### `{task.key}` · {rows[0][1].label}", "",
                  "| model | " + " | ".join(strata) + " |",
                  "|---|" + "---|" * len(strata)]
        for label, _metric, ps in rows:
            cells = []
            for s in strata:
                if s in ps:
                    value, n = ps[s][0], int(ps[s][1])
                    cells.append(f"{value:.3g}{suffix} (n={n})")
                else:
                    cells.append("—")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _disclosure_section(results: ResultSet, config: RunConfig) -> list[str]:
    """The contamination-disclosure table."""
    has_strata = any(m.per_stratum for c in results for m in c.metrics)
    d = contamination_disclosure(results, config, per_stratum=has_strata)
    lines = ["", "## Contamination disclosure", "",
             DISCLOSURE_SCHEMA_CITATION,
             "",
             "An unapplied control is coded `0`. An honest `0` beside a named mechanism "
             "is worth more than an unearned `2`.",
             "", "| field | code | basis |", "|---|---|---|"]
    lines += [f"| {name} | **{code}** | {why} |" for name, code, why in d.rows()]
    lines += ["", f"`f2_notes`: **`{d.f2_notes}`** — the five elicitation sub-elements in "
                  "order: system identity, version, token budget, attempts, attempt "
                  "resolution.",
              "", f"> {d.headline}",
              "", f"Schema: <{DISCLOSURE_SCHEMA_URL}>"]
    return lines


def _register_section(results: ResultSet, config: RunConfig) -> list[str]:
    """The Parameter Register, with inapplicable rows reasoned rather than blank."""
    rows = parameter_register(results, config)
    lines = ["", "## Parameter register", "",
             "The register from this repository's pre-Inspect pipeline, filled from this "
             "run. It assumes a locally-served quantized model, so applicability depends on "
             "the serving arrangement. A blank row and a not-applicable row make different "
             "claims, so every unfilled row states its reason.", ""]
    section = None
    for row in rows:
        if row.section != section:
            section = row.section
            lines += ["", f"### {section}", "", "| parameter | value | status |",
                      "|---|---|---|"]
        mark = {"recorded": "recorded", "not applicable": "_n/a_",
                "undisclosed": "**undisclosed**", "missing": "**missing**"}[row.status]
        lines.append(f"| `{row.parameter}` | {row.value} | {mark} |")
    return lines


def _conditions_section(results: ResultSet, config: RunConfig) -> list[str]:
    """The exact parameters every cell was run under, plus coverage and divergence."""
    c = build_conditions(results, config)
    lines = ["", "## Run conditions", "", PREAMBLE, "",
             "| parameter | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in c.shared]

    if c.all_identical:
        lines += ["", "All models were run under identical conditions for every benchmark."]
    else:
        lines += ["", "> **Conditions were NOT identical across models.** The comparison "
                      "below is not like-for-like:", ""]
        for task_key, diverged in c.divergence.items():
            for field_name, values in diverged.items():
                lines.append(f"> - `{task_key}` · **{field_name}**: {', '.join(values)}")

    lines += ["", "### Per benchmark", ""]
    for tc in c.tasks:
        lines += [f"#### `{tc.task_key}`", "", "| parameter | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in tc.rows]
        lines.append(f"| Stratum coverage | {tc.coverage_text} |")
        if tc.stratum_counts:
            detail = ", ".join(f"{k} {v}" for k, v in tc.stratum_counts.items())
            lines.append(f"| Samples per stratum | {detail} |")
        if not tc.identical_across_models:
            lines.append(f"| **Divergence** | {tc.divergence} |")
        lines.append("")

    lines += [COVERAGE_NOTE, ""]
    if c.under_covered:
        lines += ["> **Incomplete stratum coverage** — these cells did not evaluate every "
                  "category of their dataset, so their score is a score on the categories "
                  "they did reach, not on the benchmark:", ""]
        lines += [f"> - {u}" for u in c.under_covered]
    return lines


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
