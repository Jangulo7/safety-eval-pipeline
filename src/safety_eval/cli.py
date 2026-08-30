"""``safety-eval`` — the command line.

    safety-eval doctor [--metrics]        preflight: creds, datasets, gated access, drift
    safety-eval list-benchmarks           the catalog, with every metric's range and direction
    safety-eval list-models [--search]    the OpenRouter catalogue
    safety-eval run [--dry-run] ...       execute the matrix
    safety-eval report [--run-id ...]     render markdown, charts, HTML and PDF
    safety-eval gate [--run-id ...]       exit 1 on a threshold breach
    safety-eval ui                        launch the Streamlit dashboard

``run`` defaults to ``--dry-run`` being available and cheap: it resolves the whole matrix and
prints every cell it would execute, without constructing a task or opening a connection.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__


def _load_env() -> None:
    """Load ``.env`` if present. Explicit environment always wins over the file."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def _config(args: argparse.Namespace):
    from .catalog import Catalog
    from .config import RunConfig

    overrides = {
        "models": getattr(args, "models", None),
        "tasks": getattr(args, "tasks", None),
        "defaults": {
            k: getattr(args, k, None)
            for k in ("limit", "temperature", "seed", "max_connections")
        },
        "charts": _tri(getattr(args, "charts", None), getattr(args, "no_charts", False)),
        "html": _tri(None, getattr(args, "no_html", False)),
        "pdf": _tri(None, getattr(args, "no_pdf", False)),
    }
    return RunConfig.load(args.config, Catalog.load(args.catalog), overrides=overrides)


def _tri(on: bool | None, off: bool) -> bool | None:
    if off:
        return False
    return True if on else None


# ------------------------------------------------------------------------- commands

def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import diagnose

    print("\nsafety-eval preflight\n")
    d = diagnose(config_path=args.config, check_metrics=args.metrics,
                 check_network=not args.offline)
    print(d.render())
    if not d.ok:
        print("\n  Fix the failures above before running — a blocked cell still costs "
              "wall-clock and a failed matrix costs credit.\n")
    elif args.metrics:
        print("\n  Catalog verified against the installed harness. Safe to run.\n")
    else:
        print("\n  Add --metrics to also verify the catalog against a real (free, mocked) "
              "Inspect log.\n")
    return d.exit_code


def cmd_list_benchmarks(args: argparse.Namespace) -> int:
    from .catalog import Catalog, Direction

    catalog = Catalog.load(args.catalog)
    print(f"\n{len(catalog)} benchmark(s) · verified against inspect_ai "
          f"{catalog.verified_against.get('inspect_ai', '?')} / inspect_evals "
          f"{catalog.verified_against.get('inspect_evals', '?')}\n")
    for bench in catalog:
        gate = " [GATED DATASET]" if bench.gated else ""
        logs = "" if bench.publish_logs else "  [transcripts withheld]"
        print(f"  {bench.key}{gate}{logs}")
        print(f"    task     {bench.task}  (grader kwarg: {bench.grader_kwarg})")
        print(f"    measures {_fold(bench.interpretation.get('measures', ''), 74, 13)}")
        for m in bench.metrics.values():
            unit = " %" if m.unit == "percent" else ""
            direction = (
                "/".join(f"{s}:{d.value.split('_')[0]}" for s, d in m.direction_by_subset.items())
                if m.direction is Direction.CONTEXT_DEPENDENT
                else m.direction.value.split("_")[0]
            )
            star = "*" if m.primary else " "
            print(f"    {star} {m.address:<44} {m.range[0]:g}-{m.range[1]:g}{unit:<2} "
                  f"{direction} is better")
        print()
    print("  * = primary metric.  Addresses are '<score_name>/<metric_key>' as they appear "
          "in an Inspect log.\n")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    from .doctor import list_openrouter_models

    models = list_openrouter_models()
    if models is None:
        print("could not reach the OpenRouter model catalogue", file=sys.stderr)
        return 1
    needle = (args.search or "").lower()
    rows = [m for m in models if needle in m["id"].lower()
            or needle in m.get("name", "").lower()]
    rows.sort(key=lambda m: m["id"])
    print(f"\n{len(rows)} of {len(models)} OpenRouter models"
          + (f" matching {args.search!r}" if needle else "") + "\n")
    for m in rows[: args.limit]:
        pricing = m.get("pricing", {})
        prompt = _price(pricing.get("prompt"))
        completion = _price(pricing.get("completion"))
        print(f"  openrouter/{m['id']:<52} {prompt:>9} / {completion:>9} per Mtok  "
              f"ctx {m.get('context_length', '?')}")
    if len(rows) > args.limit:
        print(f"\n  ... {len(rows) - args.limit} more; narrow with --search\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .doctor import diagnose
    from .pipeline import run_matrix
    from .results import CellResult, CellStatus
    from .runner import Runner

    config = _config(args)
    runner = Runner(config)

    if args.dry_run:
        print("\n" + runner.describe_plan() + "\n")
        print("  Nothing was run and nothing was spent. Drop --dry-run to execute.\n")
        return 0

    if not args.skip_doctor:
        d = diagnose(config, check_network=not args.offline)
        if not d.ok:
            print("\npreflight failed:\n")
            print(d.render())
            print("\n  Refusing to spend on a run that will not complete. "
                  "Use --skip-doctor to override.\n")
            return 1

    total = len(config.cells)
    print(f"\nrunning {total} cells · {config.estimated_samples()} samples requested\n")
    done = 0

    def progress(cell_id: str, result: CellResult | None) -> None:
        nonlocal done
        if result is None:
            print(f"  [{done + 1:>2}/{total}] {cell_id} ... ", end="", flush=True)
            return
        done += 1
        if result.status is CellStatus.OK:
            primary = next((m for m in result.metrics if m.primary), None)
            summary = (f"{primary.label} {primary.value:.4g}"
                       f"{'%' if primary.unit == 'percent' else ''}" if primary else "ok")
            extra = (f", {sum(m.unscored_samples for m in result.metrics[:1])} unscored"
                     if any(m.unscored_samples for m in result.metrics) else "")
            print(f"{summary}{extra}  ({result.wall_clock_s:.0f}s, "
                  f"{result.total_tokens:,} tok)")
        else:
            print(f"{result.status.value.upper()}: {(result.error_message or '')[:90]}")

    results, run_dir = run_matrix(config, progress=progress)
    counts = results.status_counts()
    print(f"\n  {', '.join(f'{v} {k}' for k, v in sorted(counts.items()))} · "
          f"{results.total_tokens:,} tokens · "
          f"{results.total_wall_clock_s / 60:.1f} min")
    print(f"  results written to {run_dir / 'results.json'}\n")

    if args.report:
        return cmd_report(argparse.Namespace(**{**vars(args), "run_id": run_dir.name,
                                                "run_dir": run_dir}))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .pipeline import report

    config = _config(args)
    art = report(config, run_id=getattr(args, "run_id", None),
                 run_dir=getattr(args, "run_dir", None))

    print(f"\nreport for {art.run_dir.name}\n")
    for name, path in art.files.items():
        print(f"  {name:<12} {path}")
    for name, path in art.charts.paths.items():
        print(f"  chart:{name:<6} {path}")
    for name, why in art.skipped.items():
        print(f"  skipped {name}: {why}")

    if getattr(args, "update_readme", False):
        from .readme import ReadmeError, render_results_section, update_readme

        chart = art.charts.paths.get("calibration")
        rel = (f"results/{art.run_dir.name}/charts/calibration.png") if chart else None
        try:
            section = render_results_section(
                art.results, art.board,
                gate_passed=art.gate_report.passed,
                gate_summary=art.gate_report.summary(),
                chart_rel_path=rel,
            )
            print(f"  readme       {update_readme(Path('README.md'), section)}")
        except ReadmeError as exc:
            print(f"  readme not updated: {exc}")

    board = art.board
    if top := board.top():
        tie = f" (tied with {', '.join(top.tied_with)})" if top.tied_with else ""
        print(f"\n  {board.index_name}: {top.label} {top.index_text}{tie}")
    print(f"  gate: {'PASS' if art.gate_report.passed else 'FAIL'} — "
          f"{art.gate_report.summary()}\n")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from .gates import evaluate, render_markdown
    from .results import ResultSet, resolve_run_dir

    config = _config(args)
    run_dir = resolve_run_dir(config.output.results_dir, getattr(args, "run_id", None))
    results = ResultSet.load(run_dir / "results.json")
    report_ = evaluate(results, config)
    print()
    print(render_markdown(report_, results, config))
    return report_.exit_code


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    if not app.exists():
        print(f"dashboard not found at {app}", file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.port", str(args.port), "--server.headless", "true"]
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safety-eval",
        description="Run AISI Inspect safety benchmarks across models, compare them, and "
                    "gate a release on the result.",
    )
    p.add_argument("--version", action="version", version=f"safety-eval {__version__}")
    p.add_argument("--config", default=None, help="path to eval_config.yaml")
    p.add_argument("--catalog", default=None, help="path to benchmarks.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="preflight checks before spending anything")
    d.add_argument("--metrics", action="store_true",
                   help="also verify catalog metric names against a real (mocked, free) log")
    d.add_argument("--offline", action="store_true", help="skip all network checks")
    d.set_defaults(func=cmd_doctor)

    lb = sub.add_parser("list-benchmarks", help="the catalog, with ranges and directions")
    lb.set_defaults(func=cmd_list_benchmarks)

    lm = sub.add_parser("list-models", help="the OpenRouter model catalogue")
    lm.add_argument("--search", default="", help="substring filter on id or name")
    lm.add_argument("--limit", type=int, default=40)
    lm.set_defaults(func=cmd_list_models)

    r = sub.add_parser("run", help="execute the model x benchmark matrix")
    r.add_argument("--dry-run", action="store_true",
                   help="resolve and print every cell without running or spending anything")
    r.add_argument("--models", nargs="+", help="subset of configured model ids or labels")
    r.add_argument("--tasks", nargs="+", help="subset of configured task keys")
    r.add_argument("--limit", type=int, help="samples per task per model")
    r.add_argument("--temperature", type=float)
    r.add_argument("--seed", type=int)
    r.add_argument("--max-connections", type=int, dest="max_connections")
    r.add_argument("--skip-doctor", action="store_true", help="run without preflight checks")
    r.add_argument("--offline", action="store_true")
    r.add_argument("--report", action="store_true", help="render the report after running")
    r.add_argument("--no-charts", action="store_true")
    r.add_argument("--no-html", action="store_true")
    r.add_argument("--no-pdf", action="store_true")
    r.add_argument("--update-readme", action="store_true")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="render markdown, charts, HTML and PDF from a run")
    rep.add_argument("--run-id", default=None, help="defaults to the most recent run")
    rep.add_argument("--no-charts", action="store_true")
    rep.add_argument("--no-html", action="store_true")
    rep.add_argument("--no-pdf", action="store_true")
    rep.add_argument("--update-readme", action="store_true",
                     help="regenerate the README's results section from this run — the "
                          "only supported way to put numbers in it")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("gate", help="evaluate thresholds; exit 1 on a breach")
    g.add_argument("--run-id", default=None)
    g.set_defaults(func=cmd_gate)

    u = sub.add_parser("ui", help="launch the Streamlit dashboard")
    u.add_argument("--port", type=int, default=8501)
    u.set_defaults(func=cmd_ui)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        if os.environ.get("SAFETY_EVAL_TRACEBACK"):
            raise
        print(f"\n{type(exc).__name__}: {exc}\n"
              "(set SAFETY_EVAL_TRACEBACK=1 for the full traceback)\n", file=sys.stderr)
        return 1


def _price(value: str | None) -> str:
    try:
        per_mtok = float(value) * 1_000_000
    except (TypeError, ValueError):
        return "—"
    return "free" if per_mtok == 0 else f"${per_mtok:,.2f}"


def _fold(text: str, width: int, indent: int) -> str:
    import textwrap

    lines = textwrap.wrap(" ".join(text.split()), width=width)
    pad = " " * indent
    return ("\n" + pad).join(lines) if lines else ""


if __name__ == "__main__":
    sys.exit(main())
