"""Streamlit dashboard for the safety eval pipeline.

The point of this app is that someone who has never run an eval can pick models and
benchmarks, see what it will cost, run it, and come away understanding what the numbers
mean. So:

* the preflight runs **before** the Run button is enabled, and a gated dataset is reported
  as a fixable setup problem rather than as a mysterious failure halfway through;
* the estimated cost is shown before anything is spent, computed from OpenRouter's own
  price list;
* every metric carries its range, its direction and a plain-English explanation, all pulled
  from the same catalog the gates and the composite index read;
* the "Understand the benchmarks" tab is a first-class tab, not a tooltip.

It drives exactly the same ``pipeline.run_matrix`` / ``pipeline.report`` code as the CLI —
two entry points producing different artefacts would defeat having one source of truth.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from safety_eval.catalog import Catalog, Direction
from safety_eval.config import ConfigError, RunConfig
from safety_eval.doctor import Level, diagnose, list_openrouter_models
from safety_eval.leaderboard import build as build_leaderboard
from safety_eval.pipeline import report as build_report
from safety_eval.pipeline import run_matrix
from safety_eval.results import (
    CellStatus,
    ResultSet,
    resolve_run_dir,
)

st.set_page_config(page_title="Safety Eval Pipeline", page_icon="🛡", layout="wide")

CSS = """
<style>
  .block-container{padding-top:2.2rem;max-width:1400px}
  [data-testid="stMetricValue"]{font-size:1.55rem}
  .se-note{color:#6b6a66;font-size:.83rem;line-height:1.5}
  .se-pill{display:inline-block;font-size:.72rem;padding:2px 9px;border-radius:999px;
    border:1px solid rgba(128,128,128,.35);margin-right:6px}
  .se-ok{color:#0ca30c;border-color:#0ca30c}
  .se-fail{color:#d03b3b;border-color:#d03b3b;font-weight:600}
  .se-warn{color:#b8860b;border-color:#fab219}
  .se-card{border:1px solid rgba(128,128,128,.22);border-radius:10px;padding:14px 18px;
    margin-bottom:12px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------- resources

@st.cache_resource
def load_catalog() -> Catalog:
    return Catalog.load(ROOT / "config" / "benchmarks.yaml")


@st.cache_data(ttl=3600, show_spinner=False)
def openrouter_models() -> list[dict]:
    """The OpenRouter catalogue, so the picker can only offer models that exist."""
    return list_openrouter_models() or []


@st.cache_data(ttl=300, show_spinner=False)
def price_index() -> dict[str, tuple[float, float]]:
    """model id -> (prompt $/Mtok, completion $/Mtok), for the pre-run cost estimate."""
    out: dict[str, tuple[float, float]] = {}
    for m in openrouter_models():
        p = m.get("pricing") or {}
        try:
            out[m["id"]] = (float(p.get("prompt", 0)) * 1e6,
                            float(p.get("completion", 0)) * 1e6)
        except (TypeError, ValueError):
            continue
    return out


def base_config() -> RunConfig:
    return RunConfig.load(ROOT / "config" / "eval_config.yaml", load_catalog())


# ---------------------------------------------------------------------------- sidebar

catalog = load_catalog()
try:
    base = base_config()
    config_error = None
except (ConfigError, FileNotFoundError) as exc:
    base, config_error = None, exc

st.sidebar.title("🛡 Safety Eval")
st.sidebar.caption("AISI Inspect · OpenRouter · release gating")

if config_error:
    st.sidebar.error(f"config/eval_config.yaml is invalid:\n\n{config_error}")
    st.stop()

source = st.sidebar.radio(
    "Model source", ["OpenRouter", "AWS S3 artifact store"], horizontal=False,
    help="OpenRouter serves hosted models by id. The S3 artifact store is the original "
         "watcher: it scans a bucket for newly uploaded model artefacts and enqueues an "
         "evaluation for each one.",
)

catalogue = openrouter_models()
configured = [m.id for m in base.models]

if source == "OpenRouter":
    if catalogue:
        available = sorted(f"openrouter/{m['id']}" for m in catalogue)
        st.sidebar.caption(f"{len(available)} models available")
    else:
        available = configured
        st.sidebar.warning("OpenRouter catalogue unreachable — showing configured models "
                           "only. Model ids are not being verified.")
    default = [m for m in configured if m in available] or configured[:1]
    chosen_models = st.sidebar.multiselect("Models", options=sorted(set(available + configured)),
                                           default=default)
    grader_options = sorted(set(available + [base.grader_model]))
    grader = st.sidebar.selectbox(
        "Grader model",
        options=grader_options,
        index=grader_options.index(base.grader_model),
        help="Judge-graded metrics move when the judge moves. One grader is used for every "
             "task so the bias is at least held constant across models — its size is not "
             "measured here.",
    )
else:
    bucket = st.sidebar.text_input("S3 bucket", value=os.environ.get("S3_MODEL_BUCKET", ""))
    prefix = st.sidebar.text_input("Prefix", value="")
    st.sidebar.caption(
        "The watcher scans this bucket for new `.gguf` / `.safetensors` artefacts and "
        "enqueues an evaluation for each. Artefacts still need a serving endpoint; "
        "the eval itself runs through the configured provider."
    )
    chosen_models = st.sidebar.multiselect("Models", options=configured, default=configured)
    grader = base.grader_model

task_labels = {t.key: f"{t.key} — {catalog[t.benchmark].title.split('—')[0].strip()}"
               for t in base.tasks}
chosen_tasks = st.sidebar.multiselect(
    "Benchmarks", options=[t.key for t in base.tasks],
    default=[t.key for t in base.tasks],
    format_func=lambda k: task_labels.get(k, k),
)
gated = sorted({catalog[base.task(k).benchmark].key for k in chosen_tasks
                if catalog[base.task(k).benchmark].gated})
if gated:
    st.sidebar.caption(f"⚠ {', '.join(gated)} uses a gated dataset — needs HF_TOKEN.")

st.sidebar.divider()
limit = st.sidebar.slider("Samples per cell (n)", 5, 250, base.defaults.limit, step=5,
                          help="Capped for cost. At n=50 a two-point difference between two "
                               "models is noise, which is why every number carries an interval.")
col_a, col_b = st.sidebar.columns(2)
temperature = col_a.number_input("Temperature", 0.0, 2.0, float(base.defaults.temperature), 0.1)
seed = col_b.number_input("Seed", 0, 10_000, int(base.defaults.seed))
max_connections = st.sidebar.slider("Max connections", 1, 32, base.defaults.max_connections,
                                    help="Concurrent provider requests. Lower this if you "
                                         "are being rate-limited.")

st.sidebar.divider()
st.sidebar.caption("Outputs")
want_charts = st.sidebar.checkbox("PNG charts", value=True)
want_html = st.sidebar.checkbox("HTML leaderboard", value=True)
want_pdf = st.sidebar.checkbox("PDF report", value=True)
st.sidebar.caption(
    "<span class='se-note'>Transcripts are published only for benchmarks whose prompts and "
    "completions are benign by construction. StrongREJECT logs are never written to a "
    "published artefact.</span>", unsafe_allow_html=True)


def current_config() -> RunConfig:
    """Rebuild the config from the sidebar selections."""
    data = dict(base.raw)
    known = {m["id"]: m for m in data["models"]}
    data["models"] = [
        known.get(mid, {"id": mid, "family": mid.split("/")[1] if "/" in mid else mid,
                        "label": mid.rsplit("/", 1)[-1]})
        for mid in chosen_models
    ]
    data["grader_model"] = grader
    data["tasks"] = [t for t in data["tasks"] if t["key"] in chosen_tasks]
    data["defaults"] = {**(data.get("defaults") or {}), "limit": int(limit),
                        "temperature": float(temperature), "seed": int(seed),
                        "max_connections": int(max_connections)}
    data["output"] = {**(data.get("output") or {}), "charts": want_charts,
                      "html": want_html, "pdf": want_pdf}
    # Re-interpolate: the grader may have changed, and task args reference ${grader_model}.
    from safety_eval.config import interpolate

    raw_tasks = [dict(t) for t in base.raw["tasks"] if t["key"] in chosen_tasks]
    data["tasks"] = raw_tasks
    return RunConfig(interpolate(data), catalog)


def estimate_cost(cfg: RunConfig) -> tuple[float, str]:
    """A pre-run cost estimate from OpenRouter's own price list.

    Deliberately rough and deliberately shown *before* the Run button is enabled. The token
    assumptions are stated so the number can be argued with rather than trusted.
    """
    prices = price_index()
    if not prices:
        return math.nan, "OpenRouter prices unavailable"
    # ~350 prompt tokens and ~120 completion tokens per graded sample, plus a grader call of
    # roughly the same size. Measured from the mock runs; real tasks vary by a factor of ~2.
    per_sample_in, per_sample_out = 350, 120
    total = 0.0
    for model in cfg.models:
        key = model.id.split("/", 1)[1] if model.id.startswith("openrouter/") else model.id
        p_in, p_out = prices.get(key, (0.0, 0.0))
        n = cfg.defaults.limit * len(cfg.tasks)
        total += n * (per_sample_in * p_in + per_sample_out * p_out) / 1e6
    g_key = cfg.grader_model.split("/", 1)[1] if "/" in cfg.grader_model else cfg.grader_model
    g_in, g_out = prices.get(g_key, (0.0, 0.0))
    n_grader = cfg.defaults.limit * len(cfg.tasks) * len(cfg.models)
    total += n_grader * (per_sample_in * g_in + 60 * g_out) / 1e6
    return total, "assumes ~350 in / ~120 out tokens per sample plus one grader call each"


# ------------------------------------------------------------------------------ tabs

tab_run, tab_board, tab_charts, tab_learn, tab_prov = st.tabs(
    ["Run", "Leaderboard", "Charts", "Understand the benchmarks", "Provenance"]
)


# ---------------------------------------------------------------------------- Run tab

with tab_run:
    if not chosen_models or not chosen_tasks:
        st.info("Pick at least one model and one benchmark in the sidebar.")
        st.stop()

    try:
        cfg = current_config()
    except ConfigError as exc:
        st.error(f"Configuration is invalid: {exc}")
        st.stop()

    cells = len(cfg.cells)
    samples = cfg.estimated_samples()
    cost, cost_note = estimate_cost(cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", cells, help=f"{len(cfg.models)} models × {len(cfg.tasks)} benchmarks")
    c2.metric("Samples requested", f"{samples:,}")
    c3.metric("Estimated cost", "—" if math.isnan(cost) else f"${cost:,.2f}", help=cost_note)
    c4.metric("Grader", cfg.grader_model.rsplit("/", 1)[-1])

    st.caption(
        "The estimate is rough and stated so it can be argued with: "
        f"{cost_note}. Actual spend depends on prompt length and on how often the grader is "
        "retried."
    )

    st.subheader("Preflight")
    st.caption(
        "Run this before spending. It checks credentials, gated dataset access, model ids, "
        "and — with the deep check — whether the installed harness still emits the metric "
        "names the catalog addresses."
    )
    deep = st.checkbox("Deep check: verify catalog metric names against a real (free, "
                       "mocked) Inspect log", value=False)
    if st.button("Run preflight", type="secondary"):
        with st.spinner("checking…"):
            st.session_state["diagnosis"] = diagnose(cfg, catalog, check_metrics=deep)

    diagnosis = st.session_state.get("diagnosis")
    if diagnosis:
        for check in diagnosis.checks:
            cls = {Level.OK: "se-ok", Level.FAIL: "se-fail", Level.WARN: "se-warn",
                   Level.SKIP: ""}[check.level]
            st.markdown(
                f"<span class='se-pill {cls}'>{check.icon}</span> "
                f"<b>{check.name}</b> — {check.detail}", unsafe_allow_html=True)
            if check.fix and check.level in (Level.FAIL, Level.WARN):
                st.markdown(f"<div class='se-note' style='margin:2px 0 10px 62px'>"
                            f"{check.fix.replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True)
        if not diagnosis.ok:
            st.warning(
                f"{len(diagnosis.failures)} failure(s). Cells that cannot run are recorded "
                "as `blocked` — a fixable setup problem, kept distinct from a run error — "
                "and the rest of the matrix still completes."
            )

    st.divider()
    ready = diagnosis is not None and diagnosis.ok
    c_run, c_force = st.columns([1, 3])
    go = c_run.button("Run evaluation", type="primary", disabled=not ready)
    if not ready:
        c_force.caption("Run the preflight first. If it fails, you can still proceed — "
                        "blocked cells are recorded as such and the rest of the matrix runs.")
        if c_force.button("Run anyway", type="secondary"):
            go = True

    if go:
        progress_bar = st.progress(0.0)
        status = st.empty()
        rows_area = st.container()
        done = {"n": 0}
        started = time.time()

        def on_progress(cell_id, result):
            if result is None:
                status.markdown(f"running `{cell_id}` … ({done['n']}/{cells} done)")
                return
            done["n"] += 1
            progress_bar.progress(done["n"] / cells)
            if result.status is CellStatus.OK:
                primary = next((m for m in result.metrics if m.primary), None)
                value = (f"**{primary.label}** {primary.value:.4g}"
                         f"{'%' if primary.unit == 'percent' else ''}" if primary else "ok")
                unscored = sum(m.unscored_samples for m in result.metrics[:1])
                extra = f" · {unscored} unscored" if unscored else ""
                rows_area.markdown(
                    f"<span class='se-pill se-ok'>ok</span> `{cell_id}` — {value}{extra} "
                    f"<span class='se-note'>({result.wall_clock_s:.0f}s, "
                    f"{result.total_tokens:,} tok)</span>", unsafe_allow_html=True)
            else:
                cls = "se-warn" if result.status is CellStatus.BLOCKED else "se-fail"
                rows_area.markdown(
                    f"<span class='se-pill {cls}'>{result.status.value}</span> `{cell_id}` — "
                    f"<span class='se-note'>{(result.error_message or '')[:180]}</span>",
                    unsafe_allow_html=True)

        with st.spinner("evaluating…"):
            results, run_dir = run_matrix(cfg, progress=on_progress)
        progress_bar.progress(1.0)
        status.markdown(
            f"**done** — {', '.join(f'{v} {k}' for k, v in sorted(results.status_counts().items()))}"
            f" · {results.total_tokens:,} tokens · {(time.time() - started) / 60:.1f} min")

        with st.spinner("rendering report…"):
            art = build_report(cfg, run_dir=run_dir)
        st.session_state["run_dir"] = str(run_dir)
        st.success(f"Results written to `{run_dir}`")
        if art.gate_report.passed:
            st.success(f"Release gate PASS — {art.gate_report.summary()}")
        else:
            st.error(f"Release gate FAIL — {art.gate_report.summary()}")

    st.divider()
    st.subheader("Load an earlier run")
    runs = sorted((p.name for p in Path(cfg.output.results_dir).glob("run-*")
                   if (p / "results.json").exists()), reverse=True)
    if runs:
        pick = st.selectbox("Run", runs,
                            index=runs.index(Path(st.session_state["run_dir"]).name)
                            if st.session_state.get("run_dir")
                            and Path(st.session_state["run_dir"]).name in runs else 0)
        if st.button("Load"):
            st.session_state["run_dir"] = str(Path(cfg.output.results_dir) / pick)
            st.rerun()
    else:
        st.caption("No runs yet.")


# --------------------------------------------------------------------- loaded results

def loaded() -> tuple[ResultSet, RunConfig, Path] | None:
    cfg = current_config()
    raw = st.session_state.get("run_dir")
    try:
        run_dir = Path(raw) if raw else resolve_run_dir(cfg.output.results_dir)
        return ResultSet.load(run_dir / "results.json"), cfg, run_dir
    except (FileNotFoundError, ValueError):
        return None


with tab_board:
    state = loaded()
    if state is None:
        st.info("No results yet. Run an evaluation, or load an earlier run from the Run tab.")
    else:
        results, cfg, run_dir = state
        board = build_leaderboard(results, cfg)
        st.subheader(board.index_name)
        st.caption("Weights: " + ", ".join(f"`{k}` {v:.0%}"
                                           for k, v in board.weights.items()))

        import pandas as pd

        table = []
        for row in board.rows:
            entry = {
                "#": row.rank_text,
                "Model": row.label + ("  (tied)" if row.tied_with else ""),
                board.index_name: None if math.isnan(row.index) else round(row.index, 3),
                "95% CI": (f"[{row.interval.low:.3f}, {row.interval.high:.3f}]"
                           if row.interval.available else "—"),
            }
            for ref in board.metric_order:
                c = row.metrics.get(ref)
                entry[board.metric_labels[ref]] = c.format_value() if c else "—"
            table.append(entry)
        st.dataframe(pd.DataFrame(table), width='stretch', hide_index=True)

        for note in board.notes:
            st.markdown(f"<div class='se-note'>• {note}</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Release gate")
        from safety_eval.gates import evaluate

        gate_report = evaluate(results, cfg)
        (st.success if gate_report.passed else st.error)(
            f"{'PASS' if gate_report.passed else 'FAIL'} — {gate_report.summary()}")
        st.caption("Thresholds are illustrative defaults chosen to demonstrate the "
                   "mechanism. They are not safety claims.")
        st.dataframe(pd.DataFrame([{
            "Gate": g.gate_id, "Model": g.model_label, "Metric": g.metric_label,
            "Bound": g.bound_text, "Observed": g.observed_text,
            "Outcome": g.outcome.value, "Note": g.detail,
        } for g in gate_report.results]), width='stretch', hide_index=True)

        st.divider()
        for label, key in [("HTML leaderboard", "leaderboard.html"),
                           ("PDF report", "report.pdf"),
                           ("results.json", "results.json"),
                           ("results.md", "results.md")]:
            path = run_dir / key
            if path.exists():
                st.download_button(f"Download {label}", path.read_bytes(), file_name=key,
                                   key=f"dl-{key}")


with tab_charts:
    state = loaded()
    if state is None:
        st.info("No results yet.")
    else:
        results, cfg, run_dir = state
        captions = {
            "calibration": "A single safety score hides a trade-off between over-refusal and "
                           "under-refusal. Top right is the only good corner.",
            "leaderboard": "The composite index with 95% intervals. Overlapping whiskers "
                           "mean the ordering between those models is not supported.",
            "metric_grid": "Every metric in its own units and direction — the check on what "
                           "the composite index folded together.",
            "coverage": "An amber band is a grader that could not parse its own output; "
                        "those samples left the metric's denominator.",
        }
        found = False
        for name, caption in captions.items():
            png = run_dir / "charts" / f"{name}.png"
            if png.exists():
                found = True
                st.image(str(png), width='stretch')
                st.caption(caption)
                st.download_button(f"Download {name}.png", png.read_bytes(),
                                   file_name=f"{name}.png", key=f"png-{name}")
                st.divider()
        if not found:
            st.info("No charts for this run. Re-render them from the Run tab with "
                    "'PNG charts' enabled.")


with tab_learn:
    st.subheader("What these benchmarks actually measure")
    st.caption(
        "Ranges, directions and explanations all come from `config/benchmarks.yaml`, which "
        "is also what the release gate and the composite index read. An explanation here "
        "cannot drift away from the threshold applied to it."
    )
    st.markdown(
        "<div class='se-card'><b>Why these three together.</b> XSTest-safe and StrongREJECT "
        "measure <i>opposite</i> failure modes: a model can score perfectly on either one by "
        "being uselessly cautious or recklessly helpful. Reporting either alone is "
        "misleading. Sycophancy adds a third axis that accuracy benchmarks cannot see — "
        "whether the model keeps a correct answer when a user pushes back."
        "</div>", unsafe_allow_html=True)

    for bench in catalog:
        gated_tag = ("<span class='se-pill se-warn'>gated dataset</span>"
                     if bench.gated else "")
        logs_tag = ("<span class='se-pill se-warn'>transcripts withheld</span>"
                    if not bench.publish_logs else "")
        with st.expander(f"{bench.title}", expanded=bench.key == "xstest"):
            st.markdown(f"<span class='se-pill'>{bench.task}</span>{gated_tag}{logs_tag}",
                        unsafe_allow_html=True)
            for heading, key in [("What it measures", "measures"),
                                 ("Why it matters", "why_it_matters"),
                                 ("What it does *not* measure", "does_not_measure"),
                                 ("Reading the result", "reading_the_result")]:
                if text := bench.interpretation.get(key):
                    st.markdown(f"**{heading}**")
                    st.markdown(f"<div class='se-note'>{text}</div>", unsafe_allow_html=True)

            st.markdown("**Metrics**")
            import pandas as pd

            st.dataframe(pd.DataFrame([{
                "Metric": m.short,
                "Address in the log": m.address,
                "Range": f"{m.range[0]:g}–{m.range[1]:g}"
                         + (" %" if m.unit == "percent" else ""),
                "Direction": ("; ".join(f"{s}: {d.value.replace('_', ' ')}"
                                        for s, d in m.direction_by_subset.items())
                              if m.direction is Direction.CONTEXT_DEPENDENT
                              else m.direction.value.replace("_", " ")),
                "What it means": m.explain,
            } for m in bench.metrics.values()]), width='stretch', hide_index=True)

            if bench.caveats:
                st.markdown("**Traps**")
                for c in bench.caveats:
                    icon = "🔴" if c.severity == "high" else "🟡"
                    st.markdown(f"<div class='se-note'>{icon} <b>{c.id}</b> — {c.text}</div>",
                                unsafe_allow_html=True)

    st.divider()
    st.subheader("How to read a number from this pipeline")
    st.markdown(
        "<div class='se-note'>"
        "<b>1. Check the denominator.</b> `unscored` counts samples whose grader output "
        "could not be parsed. Those samples leave the metric entirely, so a falling XSTest "
        "refusal rate is only good news if `unscored` stayed near zero.<br><br>"
        "<b>2. Check the interval before believing an ordering.</b> At n=50 a two-point gap "
        "is noise. Models whose intervals overlap share a rank here, and that is not a "
        "hedge — it is the honest reading.<br><br>"
        "<b>3. Check the trade-off, not the total.</b> A high composite index built on a "
        "model that refuses everything is not a safe model, it is a useless one. The "
        "calibration chart is where that shows up.<br><br>"
        "<b>4. Remember the grader.</b> Every judge-graded number is a joint property of the "
        "model and the judge. Changing the grader changes the number."
        "</div>", unsafe_allow_html=True)


with tab_prov:
    state = loaded()
    if state is None:
        st.info("No results yet.")
    else:
        results, cfg, run_dir = state
        meta = results.metadata
        st.subheader("What is recorded with every score")
        st.caption(
            "A score is a joint property of the model, the harness, the grader, the prompt "
            "sample and the decoding parameters. Publishing the score alone is not "
            "reproducible, so these columns travel with it."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("inspect_ai", meta.inspect_ai_version or "—")
        c2.metric("inspect_evals", meta.inspect_evals_version or "—")
        c3.metric("Sample cap", meta.limit)
        c4.metric("Tokens", f"{results.total_tokens:,}")

        import pandas as pd

        st.dataframe(pd.DataFrame(results.to_rows()), width='stretch',
                     hide_index=True)

        if mixed := results.mixed_task_versions:
            st.warning("Mixed benchmark versions across cells — numbers from different task "
                       "versions are not directly comparable: "
                       + "; ".join(f"{k}: {sorted(v)}" for k, v in mixed.items()))

        withheld = sorted({c.task_key for c in results if not c.log_published})
        if withheld:
            st.info(
                "Aggregate scores are published for all tasks. Transcripts are published "
                f"only for benign-by-construction tasks; logs for {', '.join(withheld)} are "
                "withheld because they contain model responses to forbidden prompts. The "
                "number is the finding."
            )
