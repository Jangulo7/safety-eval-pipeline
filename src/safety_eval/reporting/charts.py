"""PNG charts for slides and the PDF report.

Four charts, each answering a question the results table cannot:

``calibration.png``
    The headline. Over-refusal against under-refusal, one point per model. A single safety
    score hides this trade-off; the chart is the argument for running both benchmarks.

``leaderboard.png``
    The composite index, ranked, with intervals. The whiskers are the point: at n=50 most
    of these bars overlap.

``metric_grid.png``
    Small multiples in native units, one panel per metric. Deliberately *not* normalised —
    it is where a reader checks whether the composite index is hiding something.

``coverage.png``
    Scored / unscored / failed samples per cell. This is the chart that catches a broken
    grader, which otherwise looks like a model that got safer.

Rules applied throughout: no cell without data is drawn as a zero; every series carries a
direct label so identity never rests on colour alone; grid and axes recede.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..config import RunConfig
from ..leaderboard import Leaderboard
from ..results import ResultSet
from . import theme


@dataclass
class ChartSet:
    """Paths of the charts that were produced, and why any were not."""

    paths: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.paths)


# Layout is computed in INCHES and converted, never in figure fractions. A fraction-based
# top margin that works on a 5in figure clips the title on a 3.7in one, which is exactly the
# kind of thing that only shows up after the chart is already in a report. Title, subtitle
# and caption are figure-level text anchored to the figure's left edge, so a chart with a
# wide y-axis label column still has its title where a reader looks for it.

_TITLE_IN = 0.30      # baseline of the title, measured down from the top edge
_SUBTITLE_IN = 0.55
_CAPTION_IN = 0.16    # bottom of the caption block, measured up from the bottom edge
_LINE_IN = 0.155      # height of one wrapped caption line at 7.5pt


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width))


def _style(fig: Figure, ax: Any) -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    fig.patch.set_facecolor(theme.SURFACE)
    ax.set_facecolor(theme.SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme.INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=theme.GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _frame(
    fig: Figure,
    *,
    title: str,
    subtitle: str = "",
    caption: str = "",
    left_in: float = 0.85,
    right_in: float = 0.22,
    bottom_in: float = 0.62,
    extra_top_in: float = 0.0,
) -> None:
    """Place the title block and caption, then size the axes around them.

    Args:
        left_in / right_in: horizontal margins in inches. ``left_in`` grows for charts whose
            y-axis carries long categorical labels.
        bottom_in: space for the x-axis label and ticks, *above* the caption block.
        extra_top_in: additional headroom between the subtitle and the axes, for charts that
            park a legend above the plot area.
    """
    width_in, height_in = fig.get_size_inches()
    caption_text = _wrap(caption, int((width_in - 0.3) * 15.8)) if caption else ""
    caption_lines = caption_text.count("\n") + 1 if caption_text else 0

    top_reserve = (_SUBTITLE_IN + 0.22 if subtitle else _TITLE_IN + 0.20) + extra_top_in
    bottom_reserve = bottom_in + (_CAPTION_IN + caption_lines * _LINE_IN if caption_text else 0.0)

    fig.subplots_adjust(
        left=left_in / width_in,
        right=1.0 - right_in / width_in,
        top=1.0 - top_reserve / height_in,
        bottom=bottom_reserve / height_in,
    )

    x = 0.16 / width_in
    fig.text(x, 1.0 - _TITLE_IN / height_in, title, color=theme.INK, fontsize=12.5,
             fontweight="bold", ha="left", va="baseline")
    if subtitle:
        fig.text(x, 1.0 - _SUBTITLE_IN / height_in, _wrap(subtitle, int(width_in * 15.5)),
                 color=theme.INK_SECONDARY, fontsize=8.5, ha="left", va="baseline",
                 linespacing=1.4)
    if caption_text:
        fig.text(x, _CAPTION_IN / height_in, caption_text, color=theme.INK_SECONDARY,
                 fontsize=7.5, ha="left", va="bottom", linespacing=1.45)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=theme.DPI, facecolor=theme.SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- calibration

def calibration_chart(results: ResultSet, config: RunConfig, path: Path) -> Path | None:
    """Over-refusal against under-refusal: the trade-off a single score hides.

    x: compliance on provably safe prompts (100 - XSTest-safe refusal_rate)
    y: refusal on forbidden prompts (1 - StrongREJECT jailbreak_rate)

    Both axes are 'more is better', so the top-right corner is the only good place to be.
    Models sitting along the diagonal have traded one failure mode for the other, which is
    not an improvement — the chart makes that visible in a way a leaderboard cannot.
    """
    tradeoff = config.catalog.tradeoffs.get("calibration")
    if tradeoff is None:
        return None

    points = []
    for i, model_id in enumerate(results.models):
        x = _tradeoff_axis(results, tradeoff["x"] if isinstance(tradeoff, dict) else tradeoff.x,
                           model_id)
        y = _tradeoff_axis(results, tradeoff.y, model_id)
        if x is None or y is None:
            continue
        points.append((results.label_for(model_id), theme.series_color(i), x, y))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    _style(fig, ax)

    ax.axhspan(0.8, 1.02, color=theme.STATUS["good"], alpha=0.045, zorder=0)

    for label, color, (x, _xlo, xhi), (y, ylo, yhi) in points:
        ax.errorbar(
            x, y,
            xerr=[[max(0.0, x - _xlo)], [max(0.0, xhi - x)]] if not math.isnan(_xlo) else None,
            yerr=[[max(0.0, y - ylo)], [max(0.0, yhi - y)]] if not math.isnan(ylo) else None,
            fmt="o", markersize=9, color=color, ecolor=color, elinewidth=1.4, capsize=3,
            markeredgecolor=theme.SURFACE, markeredgewidth=1.6, zorder=3, alpha=0.95,
        )
        # Every point is directly labelled: three of the validated hues sit under 3:1
        # against the surface, and identity must never rest on colour alone. The label is
        # anchored beside the far end of the HORIZONTAL error bar and vertically centred on
        # its own point, so a label can never sit on a whisker and its height always
        # identifies which point it belongs to. Flipping it below the marker instead would
        # let one model's label fall under another model's point.
        anchor = xhi if not math.isnan(xhi) else x
        ax.annotate(
            label, (anchor, y), textcoords="offset points", xytext=(10, 0),
            ha="left", va="center", fontsize=8.5, color=theme.INK,
            # A surface-coloured halo so a label that lands over another model's whisker
            # stays readable, rather than the two crossing each other.
            bbox={"facecolor": theme.SURFACE, "edgecolor": "none", "pad": 1.6,
                  "alpha": 0.88},
        )

    ax.set_xlabel(tradeoff.x["label"], color=theme.INK_SECONDARY, fontsize=9)
    ax.set_ylabel(tradeoff.y["label"], color=theme.INK_SECONDARY, fontsize=9)
    ax.set_xlim(-2, 124)  # right margin reserved for the direct labels
    ax.set_xticks([0, 20, 40, 60, 80, 100])  # the metric stops at 100; the margin is not data
    ax.set_ylim(-0.03, 1.06)
    ax.text(0.988, 0.972, "calibrated", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=theme.STATUS["good"], fontweight="bold")
    _frame(
        fig,
        title="Calibration: over-refusal vs under-refusal",
        subtitle="Top right is the only good corner. The diagonal is a trade, not an improvement.",
        caption=" ".join(tradeoff.caption.split())
        + f"  ·  n = {results.metadata.limit} per task per model; bars are 95% intervals.",
        left_in=0.92,
        right_in=0.28,
    )
    return _save(fig, path)


def _tradeoff_axis(results: ResultSet, axis: dict, model_id: str):
    """Resolve one axis of a trade-off pairing to (point, lo, hi), or None if unavailable."""
    task_key, address = axis["metric"].split(":", 1)
    cell = results.get(task_key, model_id)
    if cell is None or not cell.ok:
        return None
    m = cell.metric(address)
    if m is None or m.is_nan:
        return None

    transform = axis.get("transform")
    if transform == "complement_percent":
        return (100.0 - m.value,
                100.0 - m.ci_high if not math.isnan(m.ci_high) else math.nan,
                100.0 - m.ci_low if not math.isnan(m.ci_low) else math.nan)
    if transform == "complement_unit":
        return (1.0 - m.value,
                1.0 - m.ci_high if not math.isnan(m.ci_high) else math.nan,
                1.0 - m.ci_low if not math.isnan(m.ci_low) else math.nan)
    return (m.value, m.ci_low, m.ci_high)


# --------------------------------------------------------------------------- leaderboard

def leaderboard_chart(board: Leaderboard, results: ResultSet, path: Path) -> Path | None:
    """Ranked composite index with intervals. The overlap between whiskers is the finding."""
    rows = [r for r in board.rows if not math.isnan(r.index)]
    if not rows:
        return None

    order = {m: i for i, m in enumerate(results.models)}
    height = max(3.2, 0.55 * len(rows) + 2.35)
    fig, ax = plt.subplots(figsize=(7.4, height))
    _style(fig, ax)
    ax.grid(axis="y", visible=False)

    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows, strict=True):
        color = theme.series_color(order.get(row.model_id, 99))
        ax.barh(y, row.index, height=0.40, color=color, alpha=0.9, zorder=2,
                edgecolor=theme.SURFACE, linewidth=2)  # 2px surface gap between bars
        if row.interval.available:
            ax.errorbar(row.index, y,
                        xerr=[[max(0.0, row.index - row.interval.low)],
                              [max(0.0, row.interval.high - row.index)]],
                        fmt="none", ecolor=theme.INK_SECONDARY, elinewidth=1.3, capsize=3,
                        zorder=3, alpha=0.75)
        # The value label sits clear of the interval's upper cap, not of the bar end, so a
        # wide interval pushes the number out rather than colliding with its own whisker.
        label_x = max(row.index, row.interval.high if row.interval.available else row.index)
        suffix = "  (tied)" if row.tied_with else ""
        ax.text(label_x + 0.022, y, f"{row.index:.3f}{suffix}", va="center",
                fontsize=8.5, color=theme.INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.rank_text}  {r.label}" for r in rows], fontsize=9,
                       color=theme.INK)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Composite index (1 = best)", color=theme.INK_SECONDARY, fontsize=9)

    caption = ("Whiskers are 95% intervals. Models whose intervals overlap share a rank and "
               "are marked 'tied' — at this sample size an ordering between them would be a "
               "false claim.")
    _frame(
        fig,
        title=f"{board.index_name} — higher is better",
        subtitle="Weighted mean of metrics normalised to 0-1. The weighting is a choice, "
                 "not a measurement.",
        caption=caption,
        left_in=max(1.6, 0.078 * max(len(f"{r.rank_text}  {r.label}") for r in rows)),
        right_in=0.25,
        bottom_in=0.58,
    )
    return _save(fig, path)


# -------------------------------------------------------------------------- metric grid

def metric_grid_chart(results: ResultSet, config: RunConfig, path: Path) -> Path | None:
    """Small multiples in native units — the check on whether the index hides something.

    Nothing is normalised here. Each panel keeps its own scale and states its direction, so a
    reader can see that XSTest is a percentage and StrongREJECT is 0-5 rather than trusting
    that the composite handled it.
    """
    panels: list[tuple[str, Any, list[tuple[str, str, float, float, float]]]] = []
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        for spec in bench.primary_metrics or list(bench.metrics.values())[:1]:
            bars = []
            for i, model_id in enumerate(results.models):
                cell = results.get(task.key, model_id)
                if cell is None or not cell.ok:
                    continue
                m = cell.metric(spec.address)
                if m is None or m.is_nan:
                    continue
                bars.append((results.label_for(model_id), theme.series_color(i), m.value,
                             m.ci_low, m.ci_high))
            if bars:
                panels.append((task.key, spec, bars))

    if not panels:
        return None

    cols = min(2, len(panels))
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.6, 2.35 * rows_n + 1.5), squeeze=False)
    fig.patch.set_facecolor(theme.SURFACE)

    arrow = {"lower_better": "lower is better", "higher_better": "higher is better"}
    for idx, (task_key, spec, bars) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        direction = bars and next(
            (c.metric(spec.address).direction
             for c in results if c.task_key == task_key and c.ok and c.metric(spec.address)),
            spec.direction.value,
        )
        _style(fig, ax)
        # The task key alone is the panel title; the metric name, its direction and its
        # range go on the smaller line below. Putting all four on one line overflows a
        # half-width panel, and matplotlib draws the overflow straight off the canvas
        # rather than complaining.
        ax.set_title(task_key, color=theme.INK, fontsize=10.5, fontweight="bold",
                     loc="left", pad=18)
        ax.text(0.0, 1.012,
                f"{spec.short} · {arrow.get(direction, direction)} · "
                f"range {spec.range[0]:g}–{spec.range[1]:g}"
                f"{'%' if spec.unit == 'percent' else ''}",
                transform=ax.transAxes, color=theme.INK_SECONDARY, fontsize=7.5,
                va="bottom", ha="left")
        ax.grid(axis="y", visible=False)
        ys = list(range(len(bars)))[::-1]
        for y, (_label, color, value, lo, hi) in zip(ys, bars, strict=True):
            ax.barh(y, value, height=0.40, color=color, alpha=0.9, zorder=2,
                    edgecolor=theme.SURFACE, linewidth=2)
            if not math.isnan(lo):
                ax.errorbar(value, y, xerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]],
                            fmt="none", ecolor=theme.INK_SECONDARY, elinewidth=1.1,
                            capsize=2.5, zorder=3, alpha=0.7)
            label_x = max(value, hi if not math.isnan(hi) else value)
            ax.text(label_x, y, f"  {value:.3g}", va="center", fontsize=8, color=theme.INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([b[0] for b in bars], fontsize=8, color=theme.INK)
        ax.set_xlim(spec.range[0], spec.range[1] * 1.14)

    for idx in range(len(panels), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")

    width_in, height_in = fig.get_size_inches()
    fig.tight_layout(rect=(0.0, (0.16 + 2 * _LINE_IN) / height_in, 1.0,
                           1.0 - 0.62 / height_in), h_pad=3.2, w_pad=2.4)
    fig.text(0.16 / width_in, 1.0 - _TITLE_IN / height_in, "Every metric in its own units",
             fontsize=12.5, fontweight="bold", color=theme.INK, ha="left", va="baseline")
    fig.text(0.16 / width_in, 1.0 - _SUBTITLE_IN / height_in,
             "Deliberately not normalised — this is where you check what the composite index "
             "hid.", fontsize=8.5, color=theme.INK_SECONDARY, ha="left", va="baseline")
    fig.text(0.16 / width_in, _CAPTION_IN / height_in,
             _wrap("Each panel keeps its own scale and states its direction: XSTest is a "
                   "percentage, StrongREJECT is 0-5, the rest are 0-1. Bars are 95% "
                   "intervals; a model with no bar produced no value, which is not the same "
                   "as producing a zero.", int((width_in - 0.3) * 15.8)),
             fontsize=7.5, color=theme.INK_SECONDARY, ha="left", va="bottom", linespacing=1.45)
    return _save(fig, path)


# ----------------------------------------------------------------------------- coverage

def coverage_chart(results: ResultSet, path: Path) -> Path | None:
    """Scored / unscored / not-run samples per cell — the broken-grader detector.

    An unscored sample is one whose grader output could not be parsed. Those samples leave
    the metric's denominator entirely, so a cell with a large unscored band has a headline
    number that is a statement about the grader, not the model.
    """
    labels: list[str] = []
    scored: list[int] = []
    unscored: list[int] = []
    missing: list[int] = []

    for cell in results:
        requested = cell.n_requested or 0
        if cell.metrics:
            best = max(cell.metrics, key=lambda m: m.scored_samples + m.unscored_samples)
            s, u = best.scored_samples, best.unscored_samples
        else:
            s, u = 0, 0
        labels.append(f"{cell.task_key} · {cell.label}")
        scored.append(s)
        unscored.append(u)
        missing.append(max(0, requested - s - u))

    if not labels:
        return None

    height = max(3.4, 0.34 * len(labels) + 2.35)
    fig, ax = plt.subplots(figsize=(7.8, height))
    _style(fig, ax)
    ax.grid(axis="y", visible=False)

    ys = list(range(len(labels)))[::-1]
    # Status colours, never series colours: these are states, and each is labelled.
    ax.barh(ys, scored, height=0.54, color=theme.STATUS["good"], alpha=0.85,
            label="scored", edgecolor=theme.SURFACE, linewidth=2, zorder=2)
    ax.barh(ys, unscored, left=scored, height=0.54, color=theme.STATUS["warning"], alpha=0.9,
            label="unscored (grader could not parse)", edgecolor=theme.SURFACE, linewidth=2,
            zorder=2)
    ax.barh(ys, missing, left=[s + u for s, u in zip(scored, unscored, strict=True)], height=0.54,
            color=theme.STATUS["critical"], alpha=0.8, label="not scored (error / blocked)",
            edgecolor=theme.SURFACE, linewidth=2, zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5, color=theme.INK)
    ax.set_xlabel("samples", color=theme.INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, max(1, max(s + u + m for s, u, m in zip(scored, unscored, missing, strict=True))) * 1.03)
    # The legend sits above the plot rather than inside it: with one row per cell there is
    # no empty corner, and a legend box over a bar hides the very band it explains.
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=3, fontsize=8,
                       frameon=False, handlelength=1.4, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(theme.INK_SECONDARY)
    _frame(
        fig,
        title="Sample coverage per cell",
        subtitle="An unscored sample left the metric's denominator — there, the grader "
                 "failed, not the model.",
        caption="A large amber band means the headline metric for that cell describes the "
                "grader as much as the model.",
        left_in=0.16 + max(1.6, 0.056 * max(len(label) for label in labels)),
        right_in=0.25,
        bottom_in=0.55,
        extra_top_in=0.34,   # headroom for the legend parked above the plot
    )
    return _save(fig, path)


# ------------------------------------------------------------------------------ driver

def render_charts(
    results: ResultSet, config: RunConfig, board: Leaderboard, out_dir: Path
) -> ChartSet:
    """Render every chart that has data. A chart with nothing to show is skipped, not faked."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = ChartSet()

    builders = {
        "calibration": lambda p: calibration_chart(results, config, p),
        "leaderboard": lambda p: leaderboard_chart(board, results, p),
        "metric_grid": lambda p: metric_grid_chart(results, config, p),
        "coverage": lambda p: coverage_chart(results, p),
    }
    for name, build in builders.items():
        path = out_dir / f"{name}.png"
        try:
            produced = build(path)
        except Exception as exc:
            charts.skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        if produced is None:
            charts.skipped[name] = "no cell produced the data this chart needs"
        else:
            charts.paths[name] = produced
    return charts
