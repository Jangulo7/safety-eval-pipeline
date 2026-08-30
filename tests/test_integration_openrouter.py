"""Integration tests that hit a real provider.

Marked ``integration`` and excluded from CI (`-m "not integration"`). They cost money, so
they are deliberately tiny: two samples on the one benchmark whose dataset is not gated.
Their job is to prove the seam between this pipeline and the outside world, which no amount
of mocking can.

Run them with::

    pytest -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

needs_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is not set",
)
needs_hf = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="HF_TOKEN is not set; walledai/XSTest is a gated dataset",
)


@needs_key
def test_openrouter_catalogue_is_reachable_and_ids_resolve(config) -> None:
    from safety_eval.doctor import list_openrouter_models

    models = list_openrouter_models()
    assert models, "the OpenRouter model catalogue returned nothing"
    ids = {m["id"] for m in models}
    for model in config.models:
        assert model.id.split("/", 1)[1] in ids, f"{model.id} is not on OpenRouter"


@needs_key
def test_a_real_two_sample_cell_produces_a_readable_metric(config, catalog) -> None:
    """The whole seam at once: provider -> Inspect -> log -> catalog metric -> record.

    StrongREJECT because its dataset is not gated. Two samples, so the cost is negligible.
    """
    from safety_eval.config import RunConfig
    from safety_eval.results import CellStatus
    from safety_eval.runner import Runner

    scoped = RunConfig.load(None, catalog, overrides={
        "models": [config.models[0].id],
        "tasks": ["strong_reject"],
        "defaults": {"limit": 2, "max_connections": 2},
    })
    results = Runner(scoped).run()
    cell = results.cells[0]
    assert cell.status is CellStatus.OK, cell.error_message
    assert cell.full_task_version == catalog["strong_reject"].task_version_expected
    assert cell.total_tokens > 0
    assert {m.address for m in cell.metrics} == set(catalog["strong_reject"].metrics)
    assert cell.log_published is False, "StrongREJECT transcripts are never published"


@needs_key
@needs_hf
def test_xstest_runs_when_the_gated_dataset_is_accessible(config, catalog) -> None:
    from safety_eval.config import RunConfig
    from safety_eval.results import CellStatus
    from safety_eval.runner import Runner

    scoped = RunConfig.load(None, catalog, overrides={
        "models": [config.models[0].id],
        "tasks": ["xstest_safe"],
        "defaults": {"limit": 2, "max_connections": 2},
    })
    cell = Runner(scoped).run().cells[0]
    assert cell.status is CellStatus.OK, cell.error_message
    metric = cell.metric("model_graded_qa/refusal_rate")
    assert 0.0 <= metric.value <= 100.0, "refusal_rate is a percentage, not a fraction"


@needs_key
def test_without_the_gated_dataset_xstest_is_blocked_not_errored(config, catalog,
                                                                 monkeypatch) -> None:
    """The behaviour the operator actually meets on a fresh machine."""
    from safety_eval.config import RunConfig
    from safety_eval.results import CellStatus
    from safety_eval.runner import Runner

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    scoped = RunConfig.load(None, catalog, overrides={
        "models": [config.models[0].id], "tasks": ["xstest_safe"],
        "defaults": {"limit": 1},
    })
    cell = Runner(scoped).run().cells[0]
    if cell.status is CellStatus.OK:
        pytest.skip("the dataset is cached locally, so the gate cannot be exercised here")
    assert cell.status is CellStatus.BLOCKED
    assert "gated" in (cell.error_message or "").lower()
