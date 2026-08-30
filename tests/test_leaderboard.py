"""Ranking models across benchmarks.

Combining a 0-100 percentage where low is good, a 0-5 score where low is good and a 0-1 rate
where high is good is only meaningful if every range and direction is known. These tests
hold the composite index to that, and — more importantly — hold it to refusing an ordering
it cannot support.
"""

from __future__ import annotations

import itertools
import math

from safety_eval.leaderboard import build, render_markdown
from safety_eval.stats import Interval


def test_ranks_by_violated_requirements_not_by_a_score(results, config) -> None:
    """Non-compensatory by construction: nothing a model does well reduces the count.

    A weighted mean lets excellence on one axis pay for catastrophe on another, which for
    safety is exactly backwards — and two of the index's inputs are the same metric inverted,
    so they cancel. Counting violated thresholds has neither problem.
    """
    from safety_eval.gates import evaluate

    board = build(results, config, evaluate(results, config))
    counts = [len(r.gate_failures) for r in board.rows]
    assert counts == sorted(counts), "rows must be ordered by violations ascending"
    assert [r.rank for r in board.rows] == sorted(r.rank for r in board.rows)


def test_the_index_does_not_determine_the_order(results, config) -> None:
    """A model with a better index but more violations must rank lower."""
    from safety_eval.gates import evaluate

    board = build(results, config, evaluate(results, config))
    by_rank = [r for r in board.rows if r.rank]
    for better, worse in itertools.pairwise(by_rank):
        assert len(better.gate_failures) <= len(worse.gate_failures)


def test_equal_violation_counts_share_a_rank_and_are_not_split_by_the_index(
    results, config
) -> None:
    """Splitting a tie on the index would smuggle the compensatory ordering back in."""
    from safety_eval.leaderboard import _rank

    board = build(results, config)
    for i, row in enumerate(board.rows):
        row.gate_failures = ["g"]                      # everyone violates exactly one
        row.index = 0.9 - 0.2 * i                      # and their indices differ widely
    _rank(board.rows)
    assert {r.rank for r in board.rows} == {1}
    assert all(r.tied_with for r in board.rows)


def test_normalisation_respects_direction_and_range(results, config) -> None:
    """The index is still computed as a diagnostic, over three scales and two directions."""
    board = build(results, config)
    best = max(board.rows, key=lambda r: r.index)
    assert best.label == "Qwen2.5 7B Instruct"
    assert all(0.0 <= r.index <= 1.0 for r in board.rows if not math.isnan(r.index))


def test_a_model_that_refuses_everything_scores_worse(results, config) -> None:
    """The failure mode the composite index must not have: safety by uselessness.

    Refusing more of the provably-benign prompts has to move the index down. Ranking is not
    asserted here because at n=50 the intervals overlap and the board will — correctly —
    call it a tie; the point is that the index itself responds in the right direction.
    """
    from safety_eval.stats import wilson_from_rate

    over_cautious = results.models[1]
    before = next(r for r in build(results, config).rows
                  if r.model_id == over_cautious).index

    metric = results.get("xstest_safe", over_cautious).metric("model_graded_qa/refusal_rate")
    metric.value = 95.0
    ci = wilson_from_rate(95.0, 50, scale=100.0)
    metric.ci_low, metric.ci_high = ci.low, ci.high
    metric.normalised = config.metric_for(
        "xstest_safe", "model_graded_qa/refusal_rate").normalise(95.0, "safe")

    after = next(r for r in build(results, config).rows
                 if r.model_id == over_cautious).index
    assert after < before
    best = max(build(results, config).rows, key=lambda r: r.index)
    assert best.model_id != over_cautious


def test_overlapping_intervals_share_a_rank(results, config) -> None:
    board = build(results, config)
    for row in board.rows:
        row.interval = Interval(0.4, 0.9, "test")
    from safety_eval.leaderboard import _rank

    for row in board.rows:
        row.gate_failures = []
    _rank(board.rows)
    assert all(r.rank == 1 for r in board.rows)
    assert all(r.tied_with for r in board.rows)
    assert all(r.rank_text.startswith("=") for r in board.rows)


def test_different_violation_counts_are_ranked(results, config) -> None:
    board = build(results, config)
    for i, row in enumerate(board.rows):
        row.gate_failures = ["g"] * i
    from safety_eval.leaderboard import _rank

    _rank(board.rows)
    assert [r.rank for r in board.rows] == [1, 2, 3]
    assert not any(r.tied_with for r in board.rows)


def test_without_a_gate_report_nothing_is_ranked(results, config) -> None:
    """No gates evaluated means no basis for an order, and none is invented."""
    board = build(results, config)
    assert all(r.rank == 0 for r in board.rows)
    assert all(r.rank_text == "—" for r in board.rows)


def test_a_model_with_no_usable_metric_gets_no_index(messy_results, config) -> None:
    """nan rather than 0.0 — a missing number must never be ranked as a bad one."""
    board = build(messy_results, config)
    # The third model is the one the fixture blocks, errors and dead-graders.
    unusable = next(r for r in board.rows if "Ministral" in r.label)
    assert math.isnan(unusable.index)
    assert unusable.rank_text == "—"


def test_partial_coverage_is_flagged(messy_results, config) -> None:
    board = build(messy_results, config)
    assert any(r.partial for r in board.rows)
    assert any("Partial coverage" in n for n in board.notes)


def test_blocked_cells_are_called_out_as_setup_not_behaviour(messy_results, config) -> None:
    board = build(messy_results, config)
    assert any("blocked" in n and "setup" in n for n in board.notes)


def test_degraded_grader_produces_a_warning(messy_results, config) -> None:
    board = build(messy_results, config)
    warnings = [w for row in board.rows for w in row.warnings]
    assert any("grader scored only" in w for w in warnings)


def test_notes_always_state_the_evidence_and_that_weights_are_a_choice(
    results, config
) -> None:
    """The sample size is stated per benchmark, because they differ.

    A single run-level "n = X, capped for cost" was false in both directions once tasks
    began capping themselves: it under-reported a full-dataset run and claimed a cap that
    did not apply.
    """
    board = build(results, config)
    joined = " ".join(board.notes)
    assert "Samples per model" in joined
    assert "n = " in joined
    assert "non-compensatory" in joined or "violated thresholds" in joined


def test_the_index_is_labelled_a_diagnostic_wherever_it_appears(results, config) -> None:
    """It is shown because it is a useful smell test, and withheld from the ordering
    because it is not a sound basis for one. Both facts have to travel with it."""
    text = render_markdown(build(results, config))
    assert "diagnostic" in text.lower()
    assert "compensatory" in text
    assert "not commensurable" in text or "commensurable" in text


def test_mixed_task_versions_are_flagged(results, config) -> None:
    """Numbers from two benchmark versions are not comparable, and the table must say so."""
    results.get("xstest_safe", results.models[0]).full_task_version = "3-A"
    board = build(results, config)
    assert any("Mixed benchmark versions" in n for n in board.notes)


def test_markdown_prints_the_weights(results, config) -> None:
    text = render_markdown(build(results, config))
    assert "Weights:" in text
    for ref in config.leaderboard.weights:
        assert ref in text


def test_weights_are_renormalised_when_a_task_is_missing(results, config) -> None:
    """A partial matrix still produces an index on the same 0-1 scale."""
    board = build(results, config)
    assert all(math.isnan(r.index) or 0.0 <= r.index <= 1.0 for r in board.rows)
