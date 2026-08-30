"""Rendering: markdown, the HTML leaderboard, the PDF, and the charts.

The two properties that matter most here are not cosmetic:

1. **A failed cell renders as a failure**, never as a blank, a zero, or an omission. A
   partial matrix with visible gaps is honest; a filled-in one is not.
2. **No transcript from a benchmark that withholds them ever reaches an artefact.** A repo
   that publishes a working jailbreak has failed regardless of its scores.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from safety_eval.gates import evaluate
from safety_eval.leaderboard import build
from safety_eval.pipeline import report as build_report
from safety_eval.reporting.charts import render_charts
from safety_eval.reporting.html import render_leaderboard_html
from safety_eval.reporting.markdown import render_results_markdown
from safety_eval.reporting.pdf import build_pdf
from safety_eval.results import CellStatus, ResultSet


@pytest.fixture()
def rendered(messy_results, config, tmp_path):
    """Every artefact, rendered once from the messiest result set."""
    board = build(messy_results, config)
    gates = evaluate(messy_results, config)
    charts = render_charts(messy_results, config, board, tmp_path / "charts")
    html = render_leaderboard_html(messy_results, config, board, gates, charts.paths)
    pdf = build_pdf(messy_results, config, board, gates, charts.paths, tmp_path / "r.pdf")
    md = render_results_markdown(messy_results, config)
    return {"board": board, "gates": gates, "charts": charts, "html": html, "pdf": pdf,
            "md": md, "results": messy_results}


# ------------------------------------------------------------------- results.json is truth

def test_results_json_round_trips_through_strict_json(results, tmp_path) -> None:
    """Bare NaN is not valid JSON and would break every non-Python reader."""
    path = results.save(tmp_path / "results.json")
    json.loads(path.read_text())               # strict: rejects NaN / Infinity
    assert len(ResultSet.load(path)) == len(results)


def test_nan_survives_the_round_trip_as_nan(messy_results, tmp_path) -> None:
    path = messy_results.save(tmp_path / "results.json")
    back = ResultSet.load(path)
    dead = back.get("strong_reject", back.models[2])
    assert any(m.is_nan for m in dead.metrics)


def test_missing_value_reads_as_nan_not_zero(messy_results) -> None:
    """A missing number must never be plotted, ranked or gated as a good one."""
    assert math.isnan(messy_results.value("xstest_safe", messy_results.models[2],
                                          "model_graded_qa/refusal_rate"))


# ------------------------------------------------------------------------------- markdown

def test_markdown_reports_every_provenance_column(rendered) -> None:
    md = rendered["md"]
    for column in ["scored", "unscored", "version", "grader", "seed", "status"]:
        assert column in md
    assert "inspect_ai" in md and "inspect_evals" in md


def test_markdown_states_the_sample_size_per_benchmark(rendered) -> None:
    """Never one run-level number: the benchmarks differ by an order of magnitude."""
    md = rendered["md"]
    assert "Samples per model" in md
    assert "n = " in md
    assert "full dataset" in md or "capped for cost" in md


def test_markdown_explains_why_unscored_exists(rendered) -> None:
    """The column is worthless if the reader does not know what it changes."""
    md = rendered["md"]
    assert "0.3.245" in md
    assert "denominator" in md


def test_errored_and_blocked_cells_render_as_such(rendered) -> None:
    md = rendered["md"]
    assert "**blocked**" in md
    assert "**error**" in md
    assert "gated dataset" in md


def test_markdown_flags_a_degraded_grader(rendered) -> None:
    assert "grader under-performed" in rendered["md"]


# ------------------------------------------------------------------------------ artefacts

def test_html_is_self_contained(rendered) -> None:
    """It must open from a filesystem with no network: no CDN, no companion files."""
    html = rendered["html"]
    assert "http://" not in html and "https://" not in html.replace("huggingface.co", "")
    assert "data:image/png;base64," in html
    assert "<script" not in html


def test_html_carries_the_catalog_explanations(rendered, catalog) -> None:
    """The dashboard, the PDF and this page all render the same text from one source."""
    html = rendered["html"]
    for bench in catalog:
        assert bench.title in html
        assert bench.interpretation["does_not_measure"][:60] in html


def test_html_warns_about_blocked_cells_before_the_table(rendered) -> None:
    html = rendered["html"]
    assert "blocked before running" in html
    assert html.index("blocked before running") < html.index("Ranking")


def test_html_states_the_weights_and_the_limitations(rendered) -> None:
    html = rendered["html"]
    assert "not a measurement" in html
    assert "illustrative" in html
    assert "capability-controlled" in html


def test_pdf_is_produced_with_a_page_per_benchmark(rendered, catalog) -> None:
    from pypdf import PdfReader

    reader = PdfReader(str(rendered["pdf"]))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) >= 8
    for bench in catalog:
        assert bench.title.split("—")[0].strip() in text
    assert "Limitations" in text
    assert "illustrative defaults" in text


def test_pdf_leads_with_the_gate_outcome(rendered) -> None:
    from pypdf import PdfReader

    first = PdfReader(str(rendered["pdf"])).pages[0].extract_text()
    assert "Release gate" in first


# --------------------------------------------------- the non-negotiable publication rule

def test_no_withheld_transcript_reaches_any_artefact(rendered, catalog, config) -> None:
    """StrongREJECT logs contain model responses to forbidden prompts.

    The aggregate number is the finding; the completion that produced it is not something to
    distribute. This asserts the rule across every artefact at once.
    """
    from pypdf import PdfReader

    withheld = [b for b in catalog if not b.publish_logs]
    assert withheld, "the fixture must include a benchmark that withholds transcripts"

    pdf_text = "\n".join(p.extract_text() for p in PdfReader(str(rendered["pdf"])).pages)
    surfaces = [rendered["html"], rendered["md"], pdf_text]

    for cell in rendered["results"]:
        if cell.log_published:
            continue
        # The aggregate number and the log's *existence* may be reported; its contents and
        # its path may not.
        for surface in surfaces:
            assert "logs/strong_reject" not in surface
            assert (cell.log_path or "@@none@@") not in surface

    # Each artefact must state the withholding, not merely omit the content — a reader who
    # cannot see that something was deliberately held back cannot tell it apart from a gap.
    assert "transcripts withheld" in rendered["html"].lower()
    assert "Withheld transcripts" in rendered["md"]
    assert "withheld" in pdf_text.lower()


def test_records_for_withheld_tasks_carry_no_log_path(messy_results) -> None:
    for cell in messy_results:
        if not cell.log_published:
            assert cell.log_path is None


def test_the_runner_itself_withholds_the_log_path(config) -> None:
    """Asserted against the runner, not the fixture.

    The fixture set `log_path=None` for withheld tasks by construction, so the original
    test passed while the runner recorded an absolute path to every StrongREJECT log — a
    pointer to withheld content, carrying the host's filesystem layout, in a published
    artefact. A test that only exercises its own fixture proves nothing about the code.
    """
    from conftest_safety import make_log, make_score
    from safety_eval.runner import Runner

    def eval_fn(**kwargs):
        benchmark = kwargs["tasks"].split("/")[-1]
        scores = {
            "strong_reject": [make_score("strong_reject_scorer",
                                         {"jailbreak_rate": 0.1,
                                          "strong_reject_metric": 0.4})],
            "xstest": [make_score("model_graded_qa", {"refusal_rate": 12.0})],
            "sycophancy": [make_score("sycophancy_scorer",
                                      {"confidence": 0.7, "apologize_rate": 0.3})],
        }[benchmark]
        return [make_log(scores, task_version=3 if benchmark == "strong_reject" else 4,
                         full_task_version="3-A" if benchmark == "strong_reject" else "4-A")]

    results = Runner(config, run_id="run-leak", eval_fn=eval_fn).run()
    withheld = [c for c in results if not c.log_published]
    assert withheld, "the matrix must contain a task that withholds transcripts"
    for cell in withheld:
        assert cell.log_path is None, (
            f"{cell.task_key} recorded {cell.log_path!r}, a pointer to a withheld transcript"
        )
    assert any(c.log_path for c in results if c.log_published), (
        "published tasks must still record their log path"
    )


# --------------------------------------------------------------------------------- charts

def test_every_chart_is_produced_and_non_empty(rendered) -> None:
    charts = rendered["charts"]
    assert set(charts.paths) == {"calibration", "leaderboard", "metric_grid", "coverage"}
    for path in charts.paths.values():
        assert path.exists() and path.stat().st_size > 5_000


def test_charts_are_valid_pngs_at_report_resolution(rendered) -> None:
    from PIL import Image

    for path in rendered["charts"].paths.values():
        with Image.open(path) as im:
            assert im.format == "PNG"
            assert im.width >= 1200, "charts are embedded in a PDF and a report deck"


def test_a_chart_with_no_data_is_skipped_not_faked(catalog, config, tmp_path) -> None:
    """Plotting a missing cell as zero would invent a result."""
    from fixtures.factory import make_results

    results = make_results(catalog=catalog, with_blocked=True, with_dead_grader=True)
    for cell in results:
        if cell.benchmark == "strong_reject":
            cell.status = CellStatus.ERROR
            cell.metrics = []
    charts = render_charts(results, config, build(results, config), tmp_path)
    assert "calibration" in charts.skipped
    assert "no cell produced the data" in charts.skipped["calibration"]


def test_chart_failure_does_not_lose_the_others(config, results, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("safety_eval.reporting.charts.coverage_chart",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    charts = render_charts(results, config, build(results, config), tmp_path)
    assert "coverage" in charts.skipped
    assert len(charts.paths) == 3


# ------------------------------------------------------------------------------- pipeline

def test_report_pass_writes_every_artefact(config, results, tmp_path) -> None:
    run_dir = tmp_path / "run-x"
    results.save(run_dir / "results.json")
    art = build_report(config, run_dir=run_dir)
    for name in ("results_md", "gate_md", "html", "pdf"):
        assert art.files[name].exists(), f"{name} was not written: {art.skipped}"
    assert (run_dir / "charts" / "calibration.png").exists()


def test_report_is_regenerable_from_results_json_alone(config, results, tmp_path) -> None:
    """No renderer may reach past results.json to a live log or a provider.

    That is what makes 'never invent results' mechanically enforceable, and it means a
    published artefact can be audited long after the run.
    """
    run_dir = tmp_path / "run-y"
    results.save(run_dir / "results.json")
    first = build_report(config, run_dir=run_dir)
    second = build_report(config, run_dir=run_dir)
    assert first.files["gate_md"].read_text() == second.files["gate_md"].read_text()
    assert first.board.top().index == second.board.top().index


def test_report_exit_code_follows_the_gate(config, results, tmp_path) -> None:
    run_dir = tmp_path / "run-z"
    results.save(run_dir / "results.json")
    art = build_report(config, run_dir=run_dir)
    assert art.exit_code == art.gate_report.exit_code


# ------------------------------------------------------------------- the README contract

def test_readme_section_renders_from_a_real_run(results, config) -> None:
    from safety_eval.gates import evaluate
    from safety_eval.readme import render_results_section

    results.metadata.notes = ""            # a real run carries no synthetic marker
    board = build(results, config)
    gates = evaluate(results, config)
    section = render_results_section(results, board, gate_passed=gates.passed,
                                     gate_summary=gates.summary())
    assert "Samples per model" in section
    assert "n = " in section
    assert "Release gate" in section
    for row in board.rows:
        assert row.label in section


def test_readme_section_reports_the_cells_that_failed(messy_results, config) -> None:
    """A README that silently drops the cells that did not work is advertising, not reporting."""
    from safety_eval.readme import render_results_section

    messy_results.metadata.notes = ""
    section = render_results_section(
        messy_results, build(messy_results, config),
        gate_passed=False, gate_summary="x")
    assert "did not produce a number" in section
    assert "blocked" in section
    assert "gated dataset" in section


def test_readme_refuses_numbers_from_a_synthetic_run(results, config) -> None:
    """The ground rule, enforced: every number in the README comes from a committed log."""
    from safety_eval.readme import ReadmeError, render_results_section

    assert results.metadata.notes, "the fixture must mark itself synthetic"
    with pytest.raises(ReadmeError, match="refusing"):
        render_results_section(results, build(results, config),
                               gate_passed=True, gate_summary="x")


def test_readme_update_only_touches_the_marked_block(tmp_path) -> None:
    from safety_eval.readme import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nbefore\n\n<!-- RESULTS:BEGIN -->\nold\n<!-- RESULTS:END -->\n\nafter\n")
    update_readme(readme, "NEW CONTENT")
    text = readme.read_text()
    assert "before" in text and "after" in text
    assert "old" not in text and "NEW CONTENT" in text


def test_readme_ships_with_no_placeholder_numbers() -> None:
    """The spec's rule: do not ship a README with placeholder numbers or an empty table."""
    readme = Path(__file__).resolve().parents[1] / "README.md"
    body = readme.read_text().split("<!-- RESULTS:BEGIN -->")[1].split("<!-- RESULTS:END -->")[0]
    if "Not yet run" in body:
        assert "|" not in body, "an un-run README must not carry an empty results table"
    else:
        assert "Run `run-" in body, "a filled-in results section must name its run id"


def test_report_refuses_to_render_a_withheld_log_path(config, results, tmp_path) -> None:
    """The last gate before anything is written to a file someone might commit."""
    from safety_eval.pipeline import PublicationError, assert_publishable

    assert_publishable(results)  # the clean fixture must pass
    for cell in results:
        if not cell.log_published:
            cell.log_path = "/home/someone/logs/strong_reject/run.eval"
            break
    with pytest.raises(PublicationError, match="withheld"):
        assert_publishable(results)


def test_recorded_log_paths_are_repo_relative(config) -> None:
    """An absolute path carries the host's filesystem layout into a committed artefact.

    It is also meaningless to anyone who checks the repository out somewhere else, so the
    provenance it is supposed to provide is lost precisely when it would be needed.
    """
    from conftest_safety import make_log, make_score
    from safety_eval.runner import Runner

    def eval_fn(**kwargs):
        benchmark = kwargs["tasks"].split("/")[-1]
        scores = {
            "strong_reject": [make_score("strong_reject_scorer",
                                         {"jailbreak_rate": 0.1,
                                          "strong_reject_metric": 0.4})],
            "xstest": [make_score("model_graded_qa", {"refusal_rate": 12.0})],
            "sycophancy": [make_score("sycophancy_scorer",
                                      {"confidence": 0.7, "apologize_rate": 0.3})],
        }[benchmark]
        log = make_log(scores, task_version=4, full_task_version="4-A")
        log.location = str(Path.cwd() / "logs" / benchmark / "model" / "run.eval")
        return [log]

    results = Runner(config, run_id="run-rel", eval_fn=eval_fn).run()
    published = [c for c in results if c.log_published and c.log_path]
    assert published, "published tasks must record a log path"
    for cell in published:
        assert not Path(cell.log_path).is_absolute(), cell.log_path
        assert str(Path.home()) not in cell.log_path
