"""Ranking models across benchmarks.

Combining a 0-100 percentage where low is good, a 0-5 score where low is good and a 0-1 rate
where high is good is only meaningful if every range and direction is known. These tests
hold the composite index to that, and — more importantly — hold it to refusing an ordering
it cannot support.
"""

from __future__ import annotations

import math

from safety_eval.leaderboard import build, render_markdown
from safety_eval.stats import Interval


def test_ranks_by_the_composite_index(results, config) -> None:
    board = build(results, config)
    indices = [r.index for r in board.rows if not math.isnan(r.index)]
    assert indices == sorted(indices, reverse=True)


def test_normalisation_respects_direction_and_range(results, config) -> None:
    """The fixture's safest model is the one that refuses least on safe prompts, refuses most
    on unsafe ones, and has the lowest StrongREJECT score — despite those being three
    different scales pointing in two different directions."""
    board = build(results, config)
    top = board.top()
    assert top.label == "Qwen2.5 7B Instruct"
    assert 0.0 <= top.index <= 1.0


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
    assert build(results, config).top().model_id != over_cautious


def test_overlapping_intervals_share_a_rank(results, config) -> None:
    board = build(results, config)
    for row in board.rows:
        row.interval = Interval(0.4, 0.9, "test")
    from safety_eval.leaderboard import _rank

    _rank(board.rows, tie_on_ci=True)
    assert all(r.rank == 1 for r in board.rows)
    assert all(r.tied_with for r in board.rows)
    assert all(r.rank_text.startswith("=") for r in board.rows)


def test_separated_intervals_are_ranked(results, config) -> None:
    board = build(results, config)
    for i, row in enumerate(board.rows):
        row.index = 0.9 - 0.3 * i
        row.interval = Interval(row.index - 0.02, row.index + 0.02, "test")
    from safety_eval.leaderboard import _rank

    _rank(board.rows, tie_on_ci=True)
    assert [r.rank for r in board.rows] == [1, 2, 3]
    assert not any(r.tied_with for r in board.rows)


def test_tie_detection_can_be_switched_off(results, config) -> None:
    board = build(results, config)
    from safety_eval.leaderboard import _rank

    _rank(board.rows, tie_on_ci=False)
    assert [r.rank for r in board.rows] == list(range(1, len(board.rows) + 1))


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
    assert "not a measurement" in joined


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
