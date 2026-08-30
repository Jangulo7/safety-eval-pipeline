"""Run conditions, stratum coverage, and the claim that a comparison is like-for-like.

A benchmark score is only comparable across models if every model was measured the same
way, and only reproducible if a reader can see what "the same way" was. These tests hold
the pipeline to stating both, and to noticing when either fails.

The stratum-coverage machinery exists because of a measured defect: all three of these
datasets ship grouped by category, so an unshuffled sample cap evaluates one corner of the
dataset. At `limit=50` against `inspect_evals==0.18.0`, XSTest-safe covers 2 of its 10
prompt types, XSTest-unsafe 2 of 8, StrongREJECT 1 of 6, sycophancy 1 of 6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from safety_eval.config import RunConfig
from safety_eval.doctor import Level, diagnose
from safety_eval.reporting.conditions import build

ROOT = Path(__file__).resolve().parents[1]


def find(diagnosis, name: str):
    return next(c for c in diagnosis.checks if c.name == name)


def config_without(catalog, tmp_path, *, sycophancy_limit=None, **defaults):
    data = yaml.safe_load((ROOT / "config" / "eval_config.yaml").read_text())
    data["defaults"].update(defaults)
    if sycophancy_limit is not None:
        for task in data["tasks"]:
            if task["key"] == "sycophancy":
                task["limit"] = sycophancy_limit
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return RunConfig.load(path, catalog)


# --------------------------------------------------------------- the shipped configuration

def test_shipped_config_seeds_the_dataset_order(config: RunConfig) -> None:
    """Not optional. Without it a capped run takes the head of a grouped dataset."""
    assert config.defaults.sample_shuffle is not None


def test_generation_seed_and_dataset_seed_are_separate_parameters(config) -> None:
    """`seed` seeds generation and has no effect on which samples are drawn.

    Conflating the two is how a run looks reproducible while silently evaluating a
    different subset each time.
    """
    conditions = config.run_conditions()
    assert "seed" in conditions and "sample_shuffle" in conditions
    assert conditions["seed"] is not None
    assert conditions["sample_shuffle"] is not None


# ------------------------------------------------------------------------ preflight checks

def test_a_cap_without_a_dataset_seed_is_refused(catalog, tmp_path) -> None:
    """Checked on sycophancy, the one task that is capped in the shipped configuration."""
    cfg = config_without(catalog, tmp_path, sample_shuffle=None)
    check = find(diagnose(cfg, catalog, check_network=False), "coverage:sycophancy")
    assert check.level is Level.FAIL
    assert "takes the head" in check.detail
    assert "sample_shuffle" in check.fix


def test_a_cap_smaller_than_the_stratum_count_is_refused(catalog, tmp_path) -> None:
    """6 source datasets cannot be covered by 5 samples however they are shuffled."""
    cfg = config_without(catalog, tmp_path, sycophancy_limit=5)
    check = find(diagnose(cfg, catalog, check_network=False), "coverage:sycophancy")
    assert check.level is Level.FAIL
    assert "cannot cover" in check.detail


def test_a_thin_but_complete_cap_warns_rather_than_fails(catalog, tmp_path) -> None:
    cfg = config_without(catalog, tmp_path, sycophancy_limit=12)
    check = find(diagnose(cfg, catalog, check_network=False), "coverage:sycophancy")
    assert check.level is Level.WARN
    assert "per stratum" in check.detail


def test_the_shipped_cap_passes_coverage(config, catalog) -> None:
    for task in ("xstest_safe", "xstest_unsafe", "strong_reject", "sycophancy"):
        check = find(diagnose(config, catalog, check_network=False), f"coverage:{task}")
        assert check.level is Level.OK, f"{task}: {check.detail}"


def test_running_the_full_dataset_always_passes(catalog, tmp_path) -> None:
    cfg = config_without(catalog, tmp_path)
    check = find(diagnose(cfg, catalog, check_network=False), "coverage:strong_reject")
    assert check.level is Level.OK
    assert "full dataset" in check.detail


# ------------------------------------------------------------------------- the report table

def test_conditions_table_names_every_parameter(results, config) -> None:
    c = build(results, config)
    keys = {k for k, _ in c.shared}
    for expected in ["Provider", "Grader model", "Samples per cell",
                     "Epochs (samples per prompt)", "Temperature", "Generation seed",
                     "Dataset-order seed (sample_shuffle)", "Max connections",
                     "inspect_ai", "inspect_evals"]:
        assert expected in keys, f"{expected} missing from the run-conditions table"


def test_every_task_reports_its_own_arguments(results, config) -> None:
    """`-T subset=safe` is the difference between two rows of the leaderboard."""
    c = build(results, config)
    safe = next(t for t in c.tasks if t.task_key == "xstest_safe")
    rendered = dict(safe.rows)
    assert "subset=safe" in rendered["Task arguments (-T)"]
    assert "scorer_model" in rendered["Grader kwarg"]

    sr = next(t for t in c.tasks if t.task_key == "strong_reject")
    assert "judge_llm" in dict(sr.rows)["Grader kwarg"], (
        "strong_reject grades through judge_llm, not scorer_model; the report must say so"
    )


def test_identical_conditions_are_asserted_from_the_cells_not_the_config(
    results, config
) -> None:
    assert build(results, config).all_identical


def test_divergent_conditions_are_detected(results, config) -> None:
    """A mid-run edit or a resumed run must not pass as a like-for-like comparison."""
    cell = results.get("xstest_safe", results.models[1])
    cell.temperature = 0.7
    c = build(results, config)
    assert not c.all_identical
    assert "temperature" in c.divergence["xstest_safe"]


def test_divergent_grader_is_detected(results, config) -> None:
    """Judge-graded metrics move when the judge moves, so this is not a like-for-like run."""
    results.get("strong_reject", results.models[0]).grader_model = "openrouter/other/model"
    assert "grader_model" in build(results, config).divergence["strong_reject"]


def test_incomplete_stratum_coverage_is_surfaced(results, config) -> None:
    cell = results.get("xstest_safe", results.models[0])
    cell.strata_covered, cell.strata_total = 2, 10
    cell.stratum_counts = {"homonyms": 25, "figurative_language": 25}
    c = build(results, config)
    assert c.under_covered
    assert "2 of 10" in c.under_covered[0]


def test_conditions_reach_every_artefact(results, config, tmp_path) -> None:
    """Markdown, HTML and PDF all render the same table from one source."""
    from pypdf import PdfReader

    from safety_eval.gates import evaluate
    from safety_eval.leaderboard import build as build_board
    from safety_eval.reporting.html import render_leaderboard_html
    from safety_eval.reporting.markdown import render_results_markdown
    from safety_eval.reporting.pdf import build_pdf

    board, gates = build_board(results, config), evaluate(results, config)
    md = render_results_markdown(results, config)
    html = render_leaderboard_html(results, config, board, gates, {})
    pdf = build_pdf(results, config, board, gates, {}, tmp_path / "r.pdf")
    pdf_text = "\n".join(p.extract_text() for p in PdfReader(str(pdf)).pages)

    for surface, name in ((md, "markdown"), (html, "html"), (pdf_text, "pdf")):
        assert "Run conditions" in surface, f"{name} has no run-conditions section"
        assert "Dataset-order seed" in surface, f"{name} omits the sample_shuffle parameter"
        assert "Generation seed" in surface, f"{name} omits the generation seed"


def test_divergence_warns_at_the_top_of_the_html(results, config) -> None:
    """A reader must meet the caveat before the ranking, not after it."""
    from safety_eval.gates import evaluate
    from safety_eval.leaderboard import build as build_board
    from safety_eval.reporting.html import render_leaderboard_html

    results.get("xstest_safe", results.models[1]).seed = 7
    html = render_leaderboard_html(results, config, build_board(results, config),
                                   evaluate(results, config), {})
    assert "not identical across models" in html
    assert html.index("not identical across models") < html.index("Ranking")


# ---------------------------------------------------------- measured against the real data

@pytest.mark.harness
def test_unshuffled_cap_really_does_collapse_to_a_few_strata(catalog) -> None:
    """The measurement the whole mechanism is built on, re-run against the real datasets.

    If a future inspect_evals reorders or pre-shuffles these datasets this test goes green
    for the wrong reason, so it asserts the *contrast*: shuffling must strictly improve
    coverage over not shuffling.
    """
    from inspect_evals.strong_reject.strong_reject import strong_reject

    task = strong_reject(judge_llm="mockllm/model")
    key = catalog["strong_reject"].dataset["stratum_key"]
    head = {s.metadata[key] for s in list(task.dataset)[:50]}

    shuffled_task = strong_reject(judge_llm="mockllm/model")
    shuffled_task.dataset.shuffle(42)
    shuffled = {s.metadata[key] for s in list(shuffled_task.dataset)[:50]}

    assert len(shuffled) > len(head), (
        f"seeded shuffle covered {len(shuffled)} strata vs {len(head)} unshuffled"
    )
    assert len(shuffled) == catalog["strong_reject"].dataset["strata"]


@pytest.mark.harness
def test_seeded_shuffle_is_reproducible(catalog) -> None:
    """Two loads with the same seed must draw the same samples, or nothing is comparable."""
    from inspect_evals.strong_reject.strong_reject import strong_reject

    def draw(seed: int) -> list[str]:
        task = strong_reject(judge_llm="mockllm/model")
        task.dataset.shuffle(seed)
        return [str(s.id) for s in list(task.dataset)[:50]]

    assert draw(42) == draw(42)
    assert draw(42) != draw(43)
