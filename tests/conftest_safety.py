"""Shared fixtures for the Inspect-harness tests.

Kept separate from the pre-existing ``conftest.py`` (which serves the S3 watcher tests) so
that the harness swap did not have to touch the original file. Both are imported by
``tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from safety_eval.catalog import Catalog
from safety_eval.config import RunConfig


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog.load(ROOT / "config" / "benchmarks.yaml")


@pytest.fixture()
def config(catalog: Catalog) -> RunConfig:
    return RunConfig.load(ROOT / "config" / "eval_config.yaml", catalog)


@pytest.fixture()
def results(catalog: Catalog):
    """A realistic run: two of its three models genuinely breach a gate."""
    from fixtures.factory import make_results

    return make_results(catalog=catalog)


@pytest.fixture()
def passing_results(catalog: Catalog):
    """A run where every model clears every gate, so a test can introduce exactly one breach."""
    from fixtures.factory import make_results

    return make_results(catalog=catalog, profile="clean")


@pytest.fixture()
def messy_results(catalog: Catalog):
    """A result set containing every awkward case a renderer has to survive."""
    from fixtures.factory import make_results

    return make_results(catalog=catalog, with_blocked=True, with_error=True,
                        with_dead_grader=True, degraded_grader=True)


# --- fake Inspect logs ------------------------------------------------------------------
#
# Built as plain namespaces rather than real EvalLog objects so the tests exercise the
# reader's tolerance of the shape it is given, and so the suite runs with no network and no
# dataset download.

def make_score(name: str, metrics: dict[str, float], scored: int = 50, unscored: int = 0):
    return SimpleNamespace(
        name=name,
        scorer=name,
        reducer=None,
        scored_samples=scored,
        unscored_samples=unscored,
        params={},
        metrics={k: SimpleNamespace(name=k, value=v) for k, v in metrics.items()},
        metadata=None,
    )


def make_log(
    scores: list,
    *,
    status: str = "success",
    completed: int = 50,
    total: int = 50,
    task_version: int = 4,
    full_task_version: str = "4-A",
    samples: list | None = None,
    tokens: int = 20_000,
):
    return SimpleNamespace(
        status=status,
        error=None,
        location="logs/fake/fake.eval",
        eval=SimpleNamespace(task_version=task_version,
                             metadata={"full_task_version": full_task_version}),
        results=SimpleNamespace(scores=scores, completed_samples=completed,
                                total_samples=total),
        stats=SimpleNamespace(model_usage={
            "m": SimpleNamespace(input_tokens=tokens // 2, output_tokens=tokens // 2,
                                 total_tokens=tokens)}),
        samples=samples or [],
    )


def make_sample(scores: dict[str, object]):
    return SimpleNamespace(
        scores={k: SimpleNamespace(value=v, explanation=None, metadata=None)
                for k, v in scores.items()}
    )


@pytest.fixture()
def xstest_log():
    """A realistic XSTest log: a percentage metric with some samples unscored."""
    return make_log([make_score("model_graded_qa", {"refusal_rate": 12.5},
                                scored=48, unscored=2)])


@pytest.fixture()
def sycophancy_log():
    """A sycophancy log, exercising namespaced metric keys and multiple score entries."""
    samples = [make_sample({"sycophancy_scorer": {
        "original_answer": "C" if i < 40 else "I",
        "admits_mistake": "C" if i < 12 else "I",
        "confidence": 1.0 if 12 <= i < 40 else 0.0,
        "apologize_rate": 1.0 if i < 12 else 0.0,
        "truthfulness": 1.0 if i >= 12 else 0.0,
    }}) for i in range(50)]
    return make_log([
        make_score("sycophancy_scorer", {"confidence": 0.7, "apologize_rate": 0.3}),
        make_score("original_answer", {"mean": 0.8, "stderr": 0.05}),
        make_score("admits_mistake", {"mean": 0.24, "stderr": 0.06}),
        make_score("truthfulness", {"inspect_evals/truthfulness": 0.76}),
    ], samples=samples)


@pytest.fixture()
def strong_reject_log():
    """A StrongREJECT log with real per-sample dicts, so the bootstrap has values."""
    samples = []
    for i in range(50):
        refused = i >= 6
        samples.append(make_sample({"strong_reject_scorer": {
            "refusal": 1 if refused else 0,
            "convincingness": 1 if refused else 4,
            "specificity": 1 if refused else 3,
        }}))
    return make_log([make_score("strong_reject_scorer",
                                {"jailbreak_rate": 0.12, "strong_reject_metric": 0.42})],
                    task_version=3, full_task_version="3-A", samples=samples)


@pytest.fixture()
def dead_grader_log():
    """Every sample unscored: the metric is nan, which is not zero."""
    import math

    return make_log([make_score("strong_reject_scorer",
                                {"jailbreak_rate": math.nan,
                                 "strong_reject_metric": math.nan},
                                scored=0, unscored=50)],
                    task_version=3, full_task_version="3-A")
