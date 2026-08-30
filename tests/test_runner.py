"""Matrix execution and, mostly, its failure handling.

A long provider-backed run is expensive to repeat, so the runner's job is to lose as little
as possible when something goes wrong: one bad cell must not take the other eleven, a setup
problem must be distinguishable from a run problem, and a dry run must cost nothing.

Every test here injects a stub in place of ``inspect_ai.eval``, so the whole matrix, its
retries and its classification run offline with no provider and no dataset.
"""

from __future__ import annotations

import pytest

from conftest_safety import make_log, make_score
from safety_eval.config import RunConfig
from safety_eval.results import CellStatus
from safety_eval.runner import Runner, classify_failure, harness_versions


def scores_for(benchmark: str):
    return {
        "strong_reject": [make_score("strong_reject_scorer",
                                     {"jailbreak_rate": 0.1, "strong_reject_metric": 0.4})],
        "xstest": [make_score("model_graded_qa", {"refusal_rate": 15.0},
                              scored=48, unscored=2)],
        "sycophancy": [
            make_score("sycophancy_scorer", {"confidence": 0.7, "apologize_rate": 0.3}),
            make_score("original_answer", {"mean": 0.8, "stderr": 0.05}),
            make_score("truthfulness", {"inspect_evals/truthfulness": 0.7}),
        ],
    }[benchmark]


def stub_eval(**kwargs):
    """Stand-in for ``inspect_ai.eval`` that returns a plausible log for any task."""
    benchmark = kwargs["tasks"].split("/")[-1]
    version = {"strong_reject": (3, "3-A")}.get(benchmark, (4, "4-A"))
    return [make_log(scores_for(benchmark), task_version=version[0],
                     full_task_version=version[1])]


def failing_eval(message: str, *, fail_times: int = 99):
    calls = {"n": 0}

    def _eval(**kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(message)
        return stub_eval(**kwargs)

    _eval.calls = calls
    return _eval


# ------------------------------------------------------------------------------ planning

def test_plan_covers_every_model_task_pair(config: RunConfig) -> None:
    plans = Runner(config).plan()
    assert len(plans) == len(config.models) * len(config.tasks)
    assert len({p.cell_id for p in plans}) == len(plans)


def test_plan_marks_which_logs_will_be_withheld(config: RunConfig) -> None:
    plans = Runner(config).plan()
    withheld = {p.task.key for p in plans if not p.publish_logs}
    assert withheld == {"strong_reject"}


def test_dry_run_description_spends_nothing(config: RunConfig, monkeypatch) -> None:
    """`describe_plan` must not construct a task or open a connection."""
    def explode(**kwargs):
        raise AssertionError("a dry run must not call the harness")

    text = Runner(config, eval_fn=explode).describe_plan()
    assert "samples requested" in text
    assert "WITHHELD" in text
    assert "gated datasets: xstest" in text


def test_log_directories_are_separated_per_cell(config: RunConfig) -> None:
    plans = Runner(config).plan()
    assert len({p.log_dir for p in plans}) == len(plans)
    sr = [p for p in plans if p.task.key == "strong_reject"]
    assert all("strong_reject" in str(p.log_dir) for p in sr), (
        "StrongREJECT logs must live under a path that .gitignore can exclude"
    )


# ----------------------------------------------------------------------------- execution

def test_successful_run_records_metrics_and_provenance(config: RunConfig) -> None:
    results = Runner(config, run_id="run-t", eval_fn=stub_eval).run()
    assert len(results) == len(config.cells)
    assert all(c.ok for c in results)

    cell = results.get("xstest_safe", config.models[0].id)
    assert cell.metric("model_graded_qa/refusal_rate").value == 15.0
    assert cell.full_task_version == "4-A"
    assert cell.grader_model == config.grader_model
    assert cell.temperature == config.defaults.temperature
    assert cell.seed == config.defaults.seed
    assert cell.inspect_ai_version == harness_versions()["inspect_ai"]
    assert cell.total_tokens > 0


def test_direction_is_resolved_per_subset(config: RunConfig) -> None:
    """The same XSTest metric is 'lower better' on safe and 'higher better' on unsafe."""
    results = Runner(config, eval_fn=stub_eval).run()
    model = config.models[0].id
    safe = results.get("xstest_safe", model).metric("model_graded_qa/refusal_rate")
    unsafe = results.get("xstest_unsafe", model).metric("model_graded_qa/refusal_rate")
    assert safe.direction == "lower_better"
    assert unsafe.direction == "higher_better"
    assert safe.normalised == pytest.approx(1 - unsafe.normalised)


def test_log_published_flag_follows_the_catalog(config: RunConfig) -> None:
    results = Runner(config, eval_fn=stub_eval).run()
    for cell in results:
        assert cell.log_published is (cell.benchmark != "strong_reject")


# ------------------------------------------------------------------------ failure handling

def test_one_failing_cell_does_not_lose_the_others(config: RunConfig) -> None:
    """A partial matrix with visible gaps is honest; a crashed run is not."""
    target = config.models[0].id

    def selective(**kwargs):
        if kwargs["model"] == target:
            raise RuntimeError("APIStatusError: 500 internal error")
        return stub_eval(**kwargs)

    results = Runner(config, eval_fn=selective).run()
    assert len(results) == len(config.cells)
    failed = [c for c in results if c.status is CellStatus.ERROR]
    assert {c.model_id for c in failed} == {target}
    assert len(results.ok_cells) == len(config.cells) - len(config.tasks)


def test_a_gated_dataset_is_blocked_not_errored(config: RunConfig) -> None:
    """Blocked is a setup problem the operator can fix; error is a run problem.

    Reporting them the same way sends the reader looking in the wrong place, and retrying a
    blocked cell costs money for a guaranteed failure.
    """
    eval_fn = failing_eval(
        "DatasetNotFoundError: Dataset 'walledai/XSTest' is a gated dataset on the Hub"
    )
    results = Runner(config, eval_fn=eval_fn).run()
    assert all(c.status is CellStatus.BLOCKED for c in results)


def test_blocked_cells_are_not_retried(config: RunConfig) -> None:
    """One attempt each, because the outcome is certain and the retry is not free."""
    eval_fn = failing_eval("401 Client Error: Unauthorized")
    results = Runner(config, eval_fn=eval_fn).run()
    assert eval_fn.calls["n"] == len(config.cells)
    assert all(c.attempts == 1 for c in results)


def test_transient_failure_is_retried_once_then_recorded(config: RunConfig) -> None:
    eval_fn = failing_eval("Connection reset by peer", fail_times=1)
    cfg = config
    results = Runner(cfg, eval_fn=eval_fn).run()
    first = results.cells[0]
    assert first.ok and first.attempts == 2


def test_retry_gives_up_after_max_retries(config: RunConfig, monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    eval_fn = failing_eval("Connection reset by peer")
    results = Runner(config, eval_fn=eval_fn).run()
    assert all(c.status is CellStatus.ERROR for c in results)
    assert all(c.attempts == 1 + config.defaults.max_retries for c in results)


def test_an_errored_eval_log_is_recorded_as_an_error(config: RunConfig) -> None:
    """Inspect can return a log with status='error' rather than raising."""
    def erroring(**kwargs):
        log = make_log([], status="error")
        log.error = "sandbox failed to start"
        return [log]

    results = Runner(config, eval_fn=erroring).run()
    assert all(c.status is CellStatus.ERROR for c in results)
    assert "sandbox" in results.cells[0].error_message


def test_a_log_with_no_readable_metric_is_an_error_not_a_silent_pass(config) -> None:
    """A completed run whose metrics cannot be read means the catalog has drifted."""
    def wrong_scores(**kwargs):
        return [make_log([make_score("unknown_scorer", {"unknown_metric": 1.0})])]

    results = Runner(config, eval_fn=wrong_scores).run()
    assert all(c.status is CellStatus.ERROR for c in results)
    assert "doctor" in results.cells[0].error_message


def test_progress_callback_reports_start_and_finish(config: RunConfig) -> None:
    events = []
    Runner(config, eval_fn=stub_eval).run(
        progress=lambda cell_id, result: events.append((cell_id, result is None)))
    assert len(events) == 2 * len(config.cells)
    assert events[0][1] is True and events[1][1] is False


@pytest.mark.parametrize("message,expected", [
    ("Dataset 'x' is a gated dataset on the Hub", CellStatus.BLOCKED),
    ("401 Client Error: Unauthorized for url", CellStatus.BLOCKED),
    ("Please ask for access", CellStatus.BLOCKED),
    ("APIStatusError: 429 rate limit exceeded", CellStatus.ERROR),
    ("ValueError: something else entirely", CellStatus.ERROR),
])
def test_failure_classification(message: str, expected: CellStatus) -> None:
    assert classify_failure(message) is expected


def test_error_messages_are_truncated(config: RunConfig, monkeypatch) -> None:
    """A provider traceback in a results table makes it unreadable."""
    monkeypatch.setattr("time.sleep", lambda *_: None)
    results = Runner(config, eval_fn=failing_eval("x" * 5000)).run()
    assert len(results.cells[0].error_message) <= 501


def test_a_full_dataset_cell_records_the_dataset_size_not_zero(config, catalog) -> None:
    """`limit: null` means "all of it", and the record must say how many that was.

    Recording 0 made the report state "n = 0 samples" for precisely the cells with the most
    evidence behind them — the inverse of the truth.
    """
    from pathlib import Path

    import yaml

    from safety_eval.config import RunConfig

    data = yaml.safe_load(Path(config.source).read_text())
    data["defaults"]["limit"] = None
    for task in data["tasks"]:
        task.pop("limit", None)
    path = Path(config.source).parent / "_full.yaml"
    path.write_text(yaml.safe_dump(data))
    try:
        full = RunConfig.load(path, catalog)
        results = Runner(full, run_id="run-full", eval_fn=stub_eval).run()
        for cell in results:
            expected = catalog[cell.benchmark].dataset["total_samples"]
            subset = (catalog[cell.benchmark].subsets.get(
                full.task(cell.task_key).subset or "", {}) or {})
            expected = subset.get("dataset_samples", expected)
            assert cell.n_requested == expected, f"{cell.task_key} recorded {cell.n_requested}"
    finally:
        path.unlink(missing_ok=True)
