"""Run configuration validation.

Every check here exists to fail *before* a run rather than after it. A 12-cell matrix at
n=50 is 600 graded samples and a real bill; discovering a typo or a unit mistake afterwards
is an expensive way to learn it.
"""

from __future__ import annotations

import pytest
import yaml

from safety_eval.catalog import Catalog
from safety_eval.config import ConfigError, RunConfig, interpolate


def write(tmp_path, data: dict):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def minimal() -> dict:
    return {
        "provider": "openrouter",
        "models": [{"id": "openrouter/a/b", "family": "a", "label": "B"}],
        "grader_model": "openrouter/g/h",
        "defaults": {"limit": 10},
        "tasks": [{"key": "sr", "benchmark": "strong_reject",
                   "args": {"judge_llm": "${grader_model}"}}],
        "gates": [],
        "leaderboard": {"index_name": "I", "weights": {}},
    }


def test_loads_the_shipped_config(config: RunConfig) -> None:
    assert config.models and config.tasks and config.gates
    assert config.leaderboard.weights


def test_interpolation_resolves_against_the_config_root(catalog: Catalog, tmp_path) -> None:
    cfg = RunConfig.load(write(tmp_path, minimal()), catalog)
    assert cfg.task("sr").args["judge_llm"] == "openrouter/g/h"


def test_interpolation_of_an_unknown_path_is_an_error() -> None:
    """A literal '${grader_model}' passed to Inspect as a model name fails much later."""
    with pytest.raises(ConfigError, match="does not resolve"):
        interpolate({"a": "${nope.nothing}"})


def test_interpolation_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("SE_TEST_VAR", "hello")
    assert interpolate({"a": "${env.SE_TEST_VAR}"}) == {"a": "hello"}
    monkeypatch.delenv("SE_TEST_VAR")
    with pytest.raises(ConfigError, match="unset environment variable"):
        interpolate({"a": "${env.SE_TEST_VAR}"})


def test_unknown_benchmark_is_rejected(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["tasks"] = [{"key": "x", "benchmark": "not_a_benchmark"}]
    with pytest.raises(ConfigError, match="unknown benchmark"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_wrong_grader_kwarg_is_rejected(catalog: Catalog, tmp_path) -> None:
    """Inspect ignores an unexpected kwarg silently, so the config layer must not.

    `strong_reject` takes `judge_llm`. Passing `scorer_model` means the task quietly grades
    with its default judge and the run is ungoverned.
    """
    data = minimal()
    data["tasks"] = [{"key": "sr", "benchmark": "strong_reject",
                      "args": {"scorer_model": "openrouter/g/h"}}]
    with pytest.raises(ConfigError, match="judge_llm"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_duplicate_task_keys_are_rejected(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["tasks"] = data["tasks"] * 2
    with pytest.raises(ConfigError, match="duplicate task key"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_gate_bound_outside_the_metric_range_is_rejected(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["tasks"].append({"key": "xs", "benchmark": "xstest", "subset": "safe",
                          "args": {"subset": "safe", "scorer_model": "${grader_model}"}})
    data["gates"] = [{"id": "g", "task": "xs", "metric": "model_graded_qa/refusal_rate",
                      "max": 150.0}]
    with pytest.raises(ConfigError, match="can never fire"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_fraction_bound_on_a_percentage_metric_is_rejected(catalog: Catalog, tmp_path) -> None:
    """The spec's own draft had `max: 0.20` on XSTest, whose refusal_rate is 0-100.

    That bound is technically in range and can essentially never fire, which is worse than
    no gate because it looks like coverage. It is the single most expensive unit mistake
    this pipeline can make, so it is rejected by name.
    """
    data = minimal()
    data["tasks"].append({"key": "xs", "benchmark": "xstest", "subset": "safe",
                          "args": {"subset": "safe", "scorer_model": "${grader_model}"}})
    data["gates"] = [{"id": "g", "task": "xs", "metric": "model_graded_qa/refusal_rate",
                      "max": 0.20}]
    with pytest.raises(ConfigError, match="did you mean 20"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_gate_bound_inside_the_range_is_accepted(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["tasks"].append({"key": "xs", "benchmark": "xstest", "subset": "safe",
                          "args": {"subset": "safe", "scorer_model": "${grader_model}"}})
    data["gates"] = [{"id": "g", "task": "xs", "metric": "model_graded_qa/refusal_rate",
                      "max": 20.0}]
    cfg = RunConfig.load(write(tmp_path, data), catalog)
    assert cfg.gates[0].max == 20.0


def test_gate_with_no_bound_is_rejected(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["gates"] = [{"id": "g", "task": "sr",
                      "metric": "strong_reject_scorer/jailbreak_rate"}]
    with pytest.raises(ConfigError, match="neither a min nor a max"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_gate_on_an_unknown_metric_is_rejected(catalog: Catalog, tmp_path) -> None:
    data = minimal()
    data["gates"] = [{"id": "g", "task": "sr", "metric": "nope/nope", "max": 1}]
    with pytest.raises(ConfigError, match="gate 'g'"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_leaderboard_weight_on_an_unknown_reference_is_rejected(catalog, tmp_path) -> None:
    data = minimal()
    data["leaderboard"] = {"index_name": "I", "weights": {"nope:a/b": 1.0}}
    with pytest.raises(ConfigError, match="leaderboard weight"):
        RunConfig.load(write(tmp_path, data), catalog)


def test_weights_are_renormalised(config: RunConfig) -> None:
    """A partial matrix still has to produce a comparable index."""
    assert sum(config.leaderboard.normalised_weights.values()) == pytest.approx(1.0)


def test_overrides_filter_models_and_tasks(catalog: Catalog) -> None:
    cfg = RunConfig.load(None, catalog, overrides={
        "tasks": ["sycophancy"], "defaults": {"limit": 7}})
    assert [t.key for t in cfg.tasks] == ["sycophancy"]
    assert cfg.defaults.limit == 7


def test_scoping_the_run_drops_out_of_scope_gates_and_records_it(catalog: Catalog) -> None:
    """A subset run cannot evaluate a gate on a task it did not run.

    Those gates are dropped so the run is possible at all, but the drop is recorded so a
    green gate report cannot be read as full coverage.
    """
    cfg = RunConfig.load(None, catalog, overrides={"tasks": ["sycophancy"]})
    assert {g.task for g in cfg.gates} == {"sycophancy"}
    assert "over_refusal_ceiling" in cfg.dropped_gates
    assert any("xstest" in w for w in cfg.dropped_weights)
    assert sum(cfg.leaderboard.normalised_weights.values()) == pytest.approx(1.0)


def test_override_naming_something_unknown_is_an_error(catalog: Catalog) -> None:
    """Filtering to nothing would silently produce an empty matrix and a confusing success."""
    with pytest.raises(ConfigError, match="not in"):
        RunConfig.load(None, catalog, overrides={"tasks": ["no_such_task"]})


def test_cells_and_sample_estimate(config: RunConfig) -> None:
    assert len(config.cells) == len(config.models) * len(config.tasks)
    expected = sum(
        (config.limit_for(t) if config.limit_for(t) is not None
         else config.catalog[t.benchmark].subsets.get(t.subset or "", {}).get(
             "dataset_samples") or config.catalog[t.benchmark].dataset["total_samples"])
        * len(config.models)
        for t in config.tasks
    )
    assert config.estimated_samples() == expected


def test_an_explicit_limit_override_beats_a_task_limit(catalog: Catalog) -> None:
    """Otherwise --limit is silently ignored for exactly the tasks that set their own.

    That is the failure mode this pipeline audits for elsewhere: a run reporting a
    parameter it never applied.
    """
    cfg = RunConfig.load(None, catalog, overrides={"defaults": {"limit": 5}})
    for task in cfg.tasks:
        assert cfg.limit_for(task) == 5, f"{task.key} kept its own limit"


def test_a_task_may_cap_itself_independently(config: RunConfig) -> None:
    """The datasets differ by an order of magnitude; one cap cannot serve all of them.

    XSTest-safe is 250 samples and runs in full; sycophancy is 4,882 and must be capped.
    """
    assert config.limit_for(config.task("xstest_safe")) is None, "runs the full dataset"
    assert config.limit_for(config.task("sycophancy")) == 250


def test_model_slug_is_filesystem_safe(config: RunConfig) -> None:
    for model in config.models:
        assert "/" not in model.slug
