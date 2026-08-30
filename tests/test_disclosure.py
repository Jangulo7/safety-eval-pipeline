"""Contamination disclosure and the Parameter Register.

Both are computed from what the run recorded, never asserted. That is the property under
test: if the pipeline stops recording something, the code must fall, not stay high. A
disclosure framework that reports the same score regardless of the evidence measures
nothing.
"""

from __future__ import annotations

from pathlib import Path

from safety_eval.disclosure import contamination_disclosure, parameter_register

ROOT = Path(__file__).resolve().parents[1]


def disc(results, config):
    has_strata = any(m.per_stratum for c in results for m in c.metrics)
    return contamination_disclosure(results, config, per_stratum=has_strata)


def code(d, key):
    if key == "f1":
        return d.f1.code
    if key == "f2":
        return d.f2.code
    if key == "f4":
        return d.f4.code
    return next(f.code for f in d.f3 if f.key == key)


# ------------------------------------------------------------------ the achieved profile

def test_the_pipeline_earns_its_disclosure_codes(results, config) -> None:
    d = disc(results, config)
    assert code(d, "f1") == "2", "per-stratum scores with n per cell"
    assert code(d, "f2") == "2", d.f2.rationale
    assert d.f2_notes == "HYYYY"
    assert code(d, "f4") == "2", "regeneration status stated for every instrument"


def test_the_two_honest_zeros_are_not_inflated(results, config) -> None:
    """`t1` and `t4` are the codes this run has not earned, and must not claim."""
    d = disc(results, config)
    assert code(d, "t1") == "1", "public benchmarks, no decontamination check was run"
    assert code(d, "t4") == "0", "no perturbation or paraphrase-robustness testing"
    assert "should be assumed" in next(f.rationale for f in d.f3 if f.key == "t1")


def test_the_headline_states_likely_contamination(results, config) -> None:
    """The single most important sentence in the disclosure."""
    headline = disc(results, config).headline
    assert "t1 = 1" in headline
    assert "predate" in headline
    assert "would be false" in headline


def test_temporal_control_relates_dates_to_items(results, config) -> None:
    """`2` needs a cutoff stated AND related to item dates — not merely a date printed."""
    rationale = next(f.rationale for f in disc(results, config).f3 if f.key == "t3")
    assert "2023-08" in rationale and "2024-02" in rationale
    assert "predate" in rationale


def test_acquired_control_names_its_mechanism(results, config) -> None:
    """The rubric's edge rule: a vague claim spread across types inflates every rate."""
    rationale = next(f.rationale for f in disc(results, config).f3 if f.key == "t5")
    assert "generate()" in rationale
    assert "no network access" in rationale


# --------------------------------------------------- the codes fall when evidence is lost

def test_losing_the_token_budget_drops_f2(results, config) -> None:
    for cell in results:
        cell.max_tokens = None
    d = disc(results, config)
    assert d.f2_notes == "HY-YY"
    assert code(d, "f2") == "1"


def test_losing_the_stratum_scores_drops_f1(results, config) -> None:
    for cell in results:
        for m in cell.metrics:
            m.per_stratum = {}
    assert code(disc(results, config), "f1") == "1", "counts remain, scores do not"


def test_losing_everything_stratified_drops_f1_to_zero(results, config) -> None:
    for cell in results:
        cell.stratum_counts = {}
        for m in cell.metrics:
            m.per_stratum = {}
    assert code(disc(results, config), "f1") == "0"


def test_losing_the_dataset_fingerprint_drops_derivative_control(results, config) -> None:
    for cell in results:
        cell.dataset_fingerprint = None
    assert code(disc(results, config), "t2") == "1"


# -------------------------------------------------------------------- parameter register

def test_register_marks_inapplicable_rows_with_a_reason(results, config) -> None:
    """A blank row and a not-applicable row are different claims to a reader."""
    rows = parameter_register(results, config)
    na = [r for r in rows if r.status == "not applicable"]
    assert na
    for row in na:
        assert len(row.value) > 15, f"{row.parameter} is marked n/a with no reason"


def test_register_covers_every_section(results, config) -> None:
    sections = {r.section.split(" · ")[0] for r in parameter_register(results, config)}
    assert sections == {"A", "B", "C", "D", "E"}


def test_register_records_both_seeds_separately(results, config) -> None:
    """The Register has one `random_seed` field; generation and sample selection are two
    different parameters and conflating them is how a run looks reproducible while drawing
    a different subset each time."""
    params = {r.parameter for r in parameter_register(results, config)}
    assert "random_seed · generation" in params
    assert "random_seed · dataset order" in params


def test_register_reports_per_benchmark_generation_parameters(results, config) -> None:
    """Parameters are held constant within a benchmark, not across the run."""
    rows = {r.parameter: r.value for r in parameter_register(results, config)}
    assert "0.75" in rows["strong_reject · temperature"]
    assert "benchmark protocol" in rows["strong_reject · temperature"]
    assert "0.0" in rows["xstest_safe · temperature"]
    assert "pipeline choice" in rows["sycophancy · temperature"]


def test_register_records_the_serving_arrangement(results, config) -> None:
    rows = {r.parameter: r for r in parameter_register(results, config)}
    assert "vLLM" in rows["inference_backend"].value
    assert rows["quant_scheme"].status == "recorded"
    assert rows["file_hash_sha256"].status == "not applicable"


def test_register_records_the_pipeline_commit(results, config) -> None:
    """The Register's `eval_tool_commit_hash` — the audit field."""
    rows = {r.parameter: r.value for r in parameter_register(results, config)}
    assert rows["eval_tool_commit_hash"]
    assert rows["eval_tool_commit_hash"] != "unavailable"


def test_hosted_models_mark_hardware_not_applicable(results, config, catalog, tmp_path) -> None:
    """The register assumes a local model; behind an API the hardware fields are not blank,
    they are inapplicable, and the report must say which."""
    import yaml

    from safety_eval.config import RunConfig

    data = yaml.safe_load((ROOT / "config" / "eval_config.yaml").read_text())
    data.pop("serving", None)
    data["models"] = [{"id": "openrouter/openai/gpt-4.1", "family": "openai",
                       "label": "GPT-4.1"}]
    path = tmp_path / "hosted.yaml"
    path.write_text(yaml.safe_dump(data))

    rows = {r.parameter: r for r in parameter_register(results, RunConfig.load(path, catalog))}
    assert rows["gpu_model"].status == "not applicable"
    assert "provider's hardware" in rows["gpu_model"].value
    assert rows["quant_scheme"].status == "undisclosed"


# --------------------------------------------------------------------------- rendering

def test_both_tables_reach_every_artefact(results, config, tmp_path) -> None:
    from pypdf import PdfReader

    from safety_eval.gates import evaluate
    from safety_eval.leaderboard import build
    from safety_eval.reporting.html import render_leaderboard_html
    from safety_eval.reporting.markdown import render_results_markdown
    from safety_eval.reporting.pdf import build_pdf

    board, gates = build(results, config), evaluate(results, config)
    md = render_results_markdown(results, config)
    html = render_leaderboard_html(results, config, board, gates, {})
    pdf = build_pdf(results, config, board, gates, {}, tmp_path / "r.pdf")
    pdf_text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)

    for surface, name in ((md, "markdown"), (html, "html"), (pdf_text, "pdf")):
        assert "Contamination disclosure" in surface, f"{name} omits the disclosure table"
        assert "Parameter register" in surface, f"{name} omits the parameter register"
        assert "HYYYY" in surface, f"{name} omits the f2_notes slot string"
        assert "Scores by stratum" in surface, f"{name} omits the per-stratum breakdown"


def test_per_stratum_scores_name_their_strata(results, config) -> None:
    """F1 = 2 requires values a reader can read off the page, against named strata."""
    from safety_eval.reporting.markdown import render_results_markdown

    md = render_results_markdown(results, config)
    assert "homonyms" in md and "privacy_public" in md
    assert "Violence" in md
    assert "n=" in md, "per-stratum values must carry n per cell"
