"""The PDF report.

The audience is someone who has to act on the result but has not run an eval before. So the
report is not a dump of the table: it leads with the gate outcome and the trade-off chart,
and it gives **one page per benchmark** explaining what the benchmark measures, what each
metric's range and direction are, how to read the number, and what it does *not* tell you.
That section is generated from ``config/benchmarks.yaml`` — the same file the gates and the
composite index read — so the explanation and the threshold can never drift apart.

Built with ReportLab's platypus flowables rather than an HTML-to-PDF converter, because the
page breaks matter: a benchmark's explanation and its numbers must not land on opposite
pages.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ..catalog import Direction
from ..config import RunConfig
from ..gates import GateOutcome, GateReport
from ..leaderboard import Leaderboard
from ..results import CellStatus, ResultSet
from . import theme
from .html import LIMITATIONS

INK = colors.HexColor(theme.INK)
INK_2 = colors.HexColor(theme.INK_SECONDARY)
MUTED = colors.HexColor(theme.INK_MUTED)
GRID = colors.HexColor(theme.GRID)
AXIS = colors.HexColor(theme.AXIS)
GOOD = colors.HexColor(theme.STATUS["good"])
WARN = colors.HexColor(theme.STATUS["warning"])
CRIT = colors.HexColor(theme.STATUS["critical"])

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=25, textColor=INK, alignment=TA_LEFT,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=10.5, leading=15,
                                   textColor=INK_2, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, textColor=INK, spaceBefore=16,
                             spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11, leading=14, textColor=INK, spaceBefore=11,
                             spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.3, leading=13.4,
                               textColor=INK_2, spaceAfter=6),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=7.8, leading=11,
                                textColor=MUTED, spaceAfter=4),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=7.6, leading=10,
                               textColor=INK_2),
        "cellhead": ParagraphStyle("ch", parent=base["Normal"], fontSize=7.2, leading=9.4,
                                   textColor=MUTED, fontName="Helvetica-Bold"),
    }
    return s


def _table(data: list[list[Any]], widths: list[float], *, align_right: list[int] = ()) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK_2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, AXIS),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _banner(text: str, color: colors.Color, styles) -> Table:
    """A left-ruled callout — the gate outcome and the warnings ride in these."""
    p = Paragraph(text, styles["body"])
    t = Table([[p]], colWidths=[CONTENT_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f1")),
    ]))
    return t


def _image(path: Path, max_w: float = CONTENT_W) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = max_w / w
    return Image(str(path), width=max_w, height=h * scale)


def build_pdf(
    results: ResultSet,
    config: RunConfig,
    board: Leaderboard,
    gate_report: GateReport,
    chart_paths: dict[str, Path],
    out_path: Path,
) -> Path:
    """Render the full PDF report."""
    styles = _styles()
    meta = results.metadata
    story: list[Any] = []

    # ---------------------------------------------------------------- 1. summary
    story.append(Paragraph("Safety benchmark report", styles["title"]))
    story.append(Paragraph(
        f"{len(results.models)} models &times; {len(results.task_keys)} safety benchmarks "
        f"from AISI Inspect · n = {meta.limit} samples per cell",
        styles["subtitle"]))

    if meta.notes:
        story.append(_banner(f"<b>Synthetic fixture.</b> {meta.notes} These numbers are "
                             "invented and must not be cited.", CRIT, styles))
        story.append(Spacer(1, 8))

    gate_color = GOOD if gate_report.passed else CRIT
    story.append(_banner(
        f"<b>Release gate: {'PASS' if gate_report.passed else 'FAIL'}</b> — "
        f"{gate_report.summary()}. Thresholds are illustrative defaults chosen to "
        "demonstrate the mechanism; they are not safety claims.",
        gate_color, styles))
    story.append(Spacer(1, 10))

    counts = results.status_counts()
    summary_rows = [
        ["Run", meta.run_id],
        ["Finished", _pretty(meta.finished_utc or meta.started_utc)],
        ["Provider", meta.provider],
        ["Grader model", meta.grader_model or "—"],
        ["Harness", f"inspect_ai {meta.inspect_ai_version} · "
                    f"inspect_evals {meta.inspect_evals_version}"],
        ["Decoding", f"temperature {next((c.temperature for c in results), 0)} · "
                     f"seed {next((c.seed for c in results), 0)}"],
        ["Cells", ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))],
        ["Cost", f"{results.total_tokens:,} tokens · "
                 f"{results.total_wall_clock_s / 60:.1f} min wall-clock"],
    ]
    story.append(_table([["", ""]] + summary_rows, [38 * mm, CONTENT_W - 38 * mm]))
    story.append(Spacer(1, 6))

    for w in _pdf_warnings(results):
        story.append(_banner(w, WARN, styles))
        story.append(Spacer(1, 5))

    # ------------------------------------------------------------ 2. leaderboard
    story.append(Paragraph(board.index_name, styles["h1"]))
    story.append(Paragraph(
        "Weights: " + ", ".join(f"<b>{k.split(':')[0]}</b> {v:.0%}"
                                for k, v in board.weights.items())
        + ". Each metric is normalised to 0–1 (1 = better) using the range and direction "
        "recorded in the benchmark catalog. The weighting is a choice, not a measurement.",
        styles["body"]))

    head = [Paragraph(h, styles["cellhead"]) for h in
            ["#", "Model", board.index_name, "95% CI"] +
            [board.metric_labels[r].replace(" · ", "<br/>") for r in board.metric_order]]
    body = []
    for row in board.rows:
        cells = [row.rank_text, row.label + ("  (tied)" if row.tied_with else ""),
                 row.index_text,
                 f"[{row.interval.low:.3f}, {row.interval.high:.3f}]"
                 if row.interval.available else "—"]
        for ref in board.metric_order:
            c = row.metrics.get(ref)
            cells.append(c.format_value() if c else "—")
        body.append([Paragraph(str(c), styles["cell"]) for c in cells])

    n_metric_cols = len(board.metric_order)
    widths = [8 * mm, 34 * mm, 17 * mm, 24 * mm]
    widths += [(CONTENT_W - sum(widths)) / max(1, n_metric_cols)] * n_metric_cols
    story.append(_table([head] + body, widths, align_right=list(range(2, 4 + n_metric_cols))))
    story.append(Spacer(1, 6))
    for note in board.notes:
        story.append(Paragraph("• " + _strip_md(note), styles["small"]))

    # ---------------------------------------------------------------- 3. charts
    if path := chart_paths.get("calibration"):
        story.append(PageBreak())
        story.append(Paragraph("The trade-off a single score hides", styles["h1"]))
        story.append(Paragraph(
            "XSTest-safe and StrongREJECT measure opposite failure modes. A model can score "
            "perfectly on either one by being uselessly cautious or recklessly helpful, so "
            "reporting either alone is misleading. Both axes below are 'more is better'.",
            styles["body"]))
        story.append(_image(Path(path)))

    for name, title, blurb in [
        ("leaderboard", "Composite index",
         "Overlapping intervals mean the ordering between those models is not supported by "
         "this sample."),
        ("metric_grid", "Every metric in its native units",
         "Deliberately not normalised — this is the check on what the composite index "
         "folded together."),
        ("coverage", "Sample coverage",
         "An unscored sample is one whose grader output could not be parsed; it leaves the "
         "metric's denominator entirely. A large amber band means the headline number for "
         "that cell describes the grader as much as the model."),
    ]:
        if path := chart_paths.get(name):
            story.append(PageBreak())
            story.append(Paragraph(title, styles["h1"]))
            story.append(Paragraph(blurb, styles["body"]))
            story.append(_image(Path(path)))

    # ------------------------------------------------- 4. one page per benchmark
    story.append(PageBreak())
    story.append(Paragraph("Understanding the benchmarks", styles["h1"]))
    story.append(Paragraph(
        "One page per benchmark: what it measures, what each metric's range and direction "
        "are, how to read the number, and what it does not tell you. Generated from the same "
        "catalog the gates and the composite index read, so an explanation here cannot drift "
        "away from the threshold applied to it.", styles["body"]))

    seen: set[str] = set()
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        if bench.key in seen:
            continue
        seen.add(bench.key)
        story.append(PageBreak())
        story += _benchmark_page(bench, results, config, styles)

    # ----------------------------------------------------------- 5. gate detail
    story.append(PageBreak())
    story.append(Paragraph("Gate detail", styles["h1"]))
    story.append(Paragraph(
        "A gate that could not be evaluated is reported as an error and fails the build. "
        "Treating an unevaluable gate as a pass would let a broken run ship silently.",
        styles["body"]))
    head = [Paragraph(h, styles["cellhead"]) for h in
            ["Gate", "Model", "Metric", "Bound", "Observed", "Outcome", "Note"]]
    body = [[Paragraph(str(v), styles["cell"]) for v in
             [g.gate_id, g.model_label, g.metric_label, g.bound_text, g.observed_text,
              g.outcome.value.upper() if g.outcome is not GateOutcome.PASS else "pass",
              g.detail]]
            for g in gate_report.results]
    story.append(_table([head] + body,
                        [26 * mm, 25 * mm, 22 * mm, 15 * mm, 16 * mm, 14 * mm,
                         CONTENT_W - 118 * mm]))

    # ------------------------------------------------------------ 6. provenance
    story.append(PageBreak())
    story.append(Paragraph("Provenance", styles["h1"]))
    story.append(Paragraph(
        "A score is a joint property of the model, the harness, the grader, the prompt "
        "sample and the decoding parameters. Publishing the score alone is not reproducible, "
        "so these columns travel with every number in this report.", styles["body"]))
    head = [Paragraph(h, styles["cellhead"]) for h in
            ["Task", "Model", "Metric", "Value", "95% CI", "Scored", "Unscored", "Ver.",
             "Status"]]
    body = []
    for c in results:
        if c.status is not CellStatus.OK:
            body.append([Paragraph(str(v), styles["cell"]) for v in
                         [c.task_key, c.label, "—", "—", "—", "—", "—",
                          c.full_task_version or "—", c.status.value]])
            continue
        for m in c.metrics:
            suffix = "%" if m.unit == "percent" else ""
            body.append([Paragraph(str(v), styles["cell"]) for v in [
                c.task_key, c.label, m.short if hasattr(m, "short") else m.label,
                "nan" if m.is_nan else f"{m.value:.4g}{suffix}",
                "—" if math.isnan(m.ci_low) else f"[{m.ci_low:.3g}, {m.ci_high:.3g}]",
                m.scored_samples, m.unscored_samples, c.full_task_version or "—", "ok"]])
    story.append(_table([head] + body,
                        [22 * mm, 24 * mm, 26 * mm, 17 * mm, 26 * mm, 13 * mm, 15 * mm,
                         12 * mm, CONTENT_W - 155 * mm]))

    withheld = sorted({c.task_key for c in results if not c.log_published})
    if withheld:
        story.append(Spacer(1, 8))
        story.append(_banner(
            "<b>Withheld transcripts.</b> Aggregate scores are published for all tasks. "
            f"Transcripts for {', '.join(withheld)} are not: they contain model responses "
            "to forbidden prompts. The number is the finding; the completion that produced "
            "it is not something to distribute.", WARN, styles))

    # ----------------------------------------------------------- 7. limitations
    story.append(PageBreak())
    story.append(Paragraph("Limitations", styles["h1"]))
    for item in LIMITATIONS:
        story.append(Paragraph("• " + item, styles["body"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Generated by safety-eval-pipeline {meta.pipeline_version} from "
        f"results/{meta.run_id}/results.json. Every number in this report traces to that "
        "file and, through it, to an Inspect .eval log.", styles["small"]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(out_path), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN + 6 * mm,
                          title=f"Safety benchmark report — {meta.run_id}",
                          author="safety-eval-pipeline")
    frame = Frame(MARGIN, MARGIN + 6 * mm, CONTENT_W, PAGE_H - 2 * MARGIN - 6 * mm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_footer(meta.run_id))])
    doc.build(story)
    return out_path


def _benchmark_page(bench, results: ResultSet, config: RunConfig, styles) -> list[Any]:
    """One benchmark explained, followed by its own numbers."""
    out: list[Any] = [Paragraph(bench.title, styles["h1"]),
                      Paragraph(f"Inspect task <b>{bench.task}</b> · version "
                                f"{bench.task_version_expected or '—'}", styles["small"])]

    for heading, key in [
        ("What it measures", "measures"),
        ("Why it matters", "why_it_matters"),
        ("What it does <i>not</i> measure", "does_not_measure"),
        ("Reading the result", "reading_the_result"),
    ]:
        if text := bench.interpretation.get(key):
            out.append(Paragraph(heading, styles["h2"]))
            out.append(Paragraph(text, styles["body"]))

    out.append(Paragraph("Metrics", styles["h2"]))
    head = [Paragraph(h, styles["cellhead"]) for h in
            ["Metric", "Range", "Direction", "What it means"]]
    body = [[Paragraph(str(v), styles["cell"]) for v in [
        f"<b>{m.short}</b>",
        f"{m.range[0]:g}–{m.range[1]:g}{' %' if m.unit == 'percent' else ''}",
        _direction(m), m.explain]] for m in bench.metrics.values()]
    out.append(_table([head] + body, [30 * mm, 18 * mm, 26 * mm, CONTENT_W - 74 * mm]))

    if bench.caveats:
        out.append(Paragraph("Traps", styles["h2"]))
        for c in bench.caveats:
            out.append(Paragraph(f"• <b>{c.id}</b> — {c.text}", styles["small"]))

    task_keys = [t.key for t in config.tasks if t.benchmark == bench.key]
    rows = []
    for task_key in task_keys:
        for model_id in results.models:
            cell = results.get(task_key, model_id)
            if cell is None:
                continue
            if cell.status is not CellStatus.OK:
                rows.append([task_key, cell.label, "—", "—", "—",
                             f"{cell.status.value}: {(cell.error_message or '')[:70]}"])
                continue
            for m in cell.metrics:
                suffix = "%" if m.unit == "percent" else ""
                rows.append([
                    task_key, cell.label, m.label,
                    "nan" if m.is_nan else f"{m.value:.4g}{suffix}",
                    "—" if math.isnan(m.ci_low) else f"[{m.ci_low:.3g}, {m.ci_high:.3g}]",
                    f"{m.scored_samples} scored, {m.unscored_samples} unscored",
                ])
    if rows:
        out.append(Paragraph("Results for this benchmark", styles["h2"]))
        head = [Paragraph(h, styles["cellhead"]) for h in
                ["Task", "Model", "Metric", "Value", "95% CI", "Coverage"]]
        body = [[Paragraph(str(v), styles["cell"]) for v in r] for r in rows]
        out.append(_table([head] + body,
                          [24 * mm, 27 * mm, 30 * mm, 18 * mm, 26 * mm,
                           CONTENT_W - 125 * mm], align_right=[3]))

    if not bench.publish_logs:
        out.append(Spacer(1, 6))
        out.append(Paragraph(
            "<b>Transcripts for this benchmark are not published.</b> Its logs contain model "
            "responses to forbidden prompts. The aggregate number above is the finding.",
            styles["small"]))
    return out


def _direction(metric) -> str:
    if metric.direction is Direction.CONTEXT_DEPENDENT:
        return "; ".join(f"{s}: {d.value.replace('_', ' ')}"
                         for s, d in metric.direction_by_subset.items())
    return metric.direction.value.replace("_", " ")


def _pdf_warnings(results: ResultSet) -> list[str]:
    out = []
    if blocked := [c for c in results if c.status is CellStatus.BLOCKED]:
        out.append(
            f"<b>{len(blocked)} cell(s) blocked before running</b> "
            f"({', '.join(sorted({c.task_key for c in blocked}))}). This is a setup problem "
            "— a gated dataset or a missing credential — not model behaviour. "
            "Run <font face='Courier'>safety-eval doctor</font> for the fix.")
    if errored := [c for c in results if c.status is CellStatus.ERROR]:
        out.append(
            f"<b>{len(errored)} cell(s) failed to run.</b> The matrix has gaps. A partial "
            "result with visible gaps is reported rather than a filled-in one.")
    if mixed := results.mixed_task_versions:
        out.append(
            "<b>Mixed benchmark versions across cells</b> ("
            + "; ".join(f"{k}: {sorted(v)}" for k, v in mixed.items())
            + "). Numbers from different task versions are not directly comparable.")
    return out


def _footer(run_id: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, MARGIN - 2 * mm,
                          f"safety-eval-pipeline · {run_id} · every number traces to "
                          "results.json")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 2 * mm, f"page {doc.page}")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, MARGIN + 2 * mm, PAGE_W - MARGIN, MARGIN + 2 * mm)
        canvas.restoreState()
    return draw


def _strip_md(text: str) -> str:
    """Markdown bold/code into ReportLab's mini-HTML."""
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def _pretty(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso
