"""Contamination disclosure and the Parameter Register.

Both are computed from what the run recorded, never asserted. That is the property under
test: if the pipeline stops recording something, the code must fall, not stay high. A
disclosure framework that reports the same score regardless of the evidence measures
nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    assert "Assume direct contamination" in next(f.rationale for f in d.f3 if f.key == "t1")


def test_the_headline_states_likely_contamination(results, config) -> None:
    """The single most important sentence in the disclosure."""
    headline = disc(results, config).headline
    assert "t1 = 1" in headline
    assert "predate" in headline
    assert "rule out any claim of an uncontaminated result" in headline


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


# ------------------------------------------------------------ provenance of every field

def test_no_register_row_is_a_placeholder(results, config) -> None:
    """The failure this vocabulary exists to prevent.

    Two rows once printed the literal string "recorded at run time" while claiming status
    `recorded`. A placeholder presented as a measurement is worse than an empty field: it
    claims evidence that does not exist.
    """
    for row in parameter_register(results, config):
        assert "recorded at run time" not in row.value, row.parameter
        assert row.value.strip(), f"{row.parameter} has an empty value"
        assert row.provenance in {"measured", "applied", "requested", "editorial",
                                  "unavailable", "n/a"}, row.provenance


def test_a_field_that_cannot_be_obtained_says_so(results, config) -> None:
    rows = {r.parameter: r for r in parameter_register(results, config)}
    unavailable = [r for r in rows.values() if r.provenance == "unavailable"]
    assert unavailable, "the register should admit what it cannot obtain"
    for row in unavailable:
        assert row.status == "missing"
        assert "not " in row.value.lower(), f"{row.parameter} must state what is absent"


def test_the_commit_reported_is_the_one_the_run_happened_at(results, config) -> None:
    """`git rev-parse HEAD` at report time answers a different question.

    The register previously reported the commit current when the report was rendered, which
    for the published run was three commits after the run itself.
    """
    for cell in results:
        cell.run_commit, cell.run_commit_dirty = "abc1234", True
    row = next(r for r in parameter_register(results, config)
               if r.parameter == "eval_tool_commit_hash")
    assert "abc1234" in row.value
    assert "DIRTY" in row.value, "a dirty tree must travel with the commit"
    assert row.provenance == "measured"


def test_generation_parameters_come_from_the_log_not_the_catalog(results, config) -> None:
    for cell in results:
        if cell.benchmark == "strong_reject":
            cell.applied_generate_config = {"temperature": 0.5, "max_tokens": 2048}
    row = next(r for r in parameter_register(results, config)
               if r.parameter == "strong_reject · temperature")
    assert "0.5" in row.value and "applied" in row.value
    assert "NOT APPLIED" in row.value, "a divergence from the declared value must be flagged"
    assert row.provenance == "measured"


def test_a_value_the_server_never_confirmed_is_marked_requested(results, config) -> None:
    for cell in results:
        cell.serving = {}
    row = next(r for r in parameter_register(results, config)
               if r.parameter == "quant_scheme")
    assert row.provenance == "requested"
    assert "did not report it back" in row.value


def test_a_value_the_server_confirmed_is_marked_measured(results, config) -> None:
    for cell in results:
        cell.serving = {"dtype": "torch.bfloat16", "quantization": "None",
                        "engine_version": "0.28.0", "max_model_len": "8192"}
    rows = {r.parameter: r for r in parameter_register(results, config)}
    assert rows["quant_scheme"].provenance == "measured"
    assert rows["inference_backend"].provenance == "measured"
    assert "0.28.0" in rows["inference_backend"].value


def test_serving_log_parser_reads_a_real_vllm_startup_line(tmp_path) -> None:
    """Verified against a line captured from vLLM 0.28.0, not against a guess."""
    from safety_eval.local_runner import query_serving

    log = tmp_path / "vllm.log"
    log.write_text(
        "(EngineCore pid=20636) INFO [core.py:122] Initializing a V1 LLM engine (v0.28.0) "
        "with config: model='Qwen/Qwen2.5-7B-Instruct', dtype=torch.bfloat16, "
        "max_seq_len=8192, quantization=None, seed=42\n")
    facts = query_serving("http://127.0.0.1:59999/v1", "k", log)
    assert facts["dtype"] == "torch.bfloat16"
    assert facts["quantization"] == "None"
    assert facts["engine_version"] == "0.28.0"


def test_published_artefacts_agree_with_results_json(config) -> None:
    """Every number in a published artefact must trace to the record it renders from.

    This ran as a throwaway script three times and was lost to /tmp twice. It belongs in the
    suite: the property it checks — that the artefacts and the record cannot drift apart —
    is one the pipeline claims on every page.
    """
    from pypdf import PdfReader

    from safety_eval.gates import evaluate
    from safety_eval.leaderboard import build
    from safety_eval.results import ResultSet

    published = Path(__file__).resolve().parents[1] / "results" / "published"
    if not (published / "results.json").exists():
        pytest.skip("no published run in this checkout")

    results = ResultSet.load(published / "results.json")
    md = (published / "results.md").read_text()
    html = (published / "leaderboard.html").read_text()
    pdf = "\n".join(p.extract_text() for p in PdfReader(str(published / "report.pdf")).pages)

    # every primary metric value appears in the markdown
    for cell in results:
        for m in cell.metrics:
            if m.primary and not m.is_nan:
                assert f"{m.value:.4g}" in md, f"{cell.task_key}/{cell.label} {m.address}"

    # the index recomputes exactly from the raw values
    board = build(results, config, evaluate(results, config))
    weights = config.leaderboard.normalised_weights
    for row in board.rows:
        total = 0.0
        for ref, w in weights.items():
            task, addr = ref.split(":", 1)
            m = results.get(task, row.model_id).metric(addr)
            scaled = (m.value - m.range_low) / (m.range_high - m.range_low)
            total += w * (scaled if m.direction == "higher_better" else 1 - scaled)
        assert abs(total - row.index) < 1e-6, row.label

    # per-stratum values weight-average back to their aggregate
    for cell in results:
        for m in cell.metrics:
            if not m.per_stratum or m.is_nan:
                continue
            pairs = list(m.per_stratum.values())
            n = sum(int(x[1]) for x in pairs)
            assert n == m.scored_samples, f"{cell.task_key} stratum n sums to {n}"
            mean = sum(x[0] * x[1] for x in pairs) / n
            assert abs(mean - m.value) < 0.6, f"{cell.task_key} {m.address}"

    # nothing withheld, and no host paths, in any artefact
    for surface, name in ((md, "md"), (html, "html"), (pdf, "pdf")):
        assert "logs/strong_reject" not in surface, name
        assert "/home/" not in surface, name


def test_a_local_run_reports_its_own_hardware(results, config) -> None:
    """Regression: a loop variable named `local` inside the inference section shadowed the
    serving flag, which silently flipped the environment section to the hosted branch and
    reported "the provider's hardware" for models served on this machine."""
    results.metadata.host = {"gpu_model": "TEST GPU", "driver_version": "1.2.3"}
    rows = {r.parameter: r for r in parameter_register(results, config)
            if r.section.startswith("C")}
    assert "TEST GPU" in rows["gpu_model"].value
    assert rows["gpu_model"].provenance == "measured"
    assert "provider's hardware" not in rows["gpu_model"].value


def test_post_hoc_host_capture_says_so(results, config) -> None:
    """Accurate and measured-during-the-run are different claims."""
    results.metadata.host = {"gpu_model": "TEST GPU", "captured": "after the run"}
    row = next(r for r in parameter_register(results, config) if r.parameter == "gpu_model")
    assert "after the run" in row.value
