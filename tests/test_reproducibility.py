"""The reproducibility gate.

`doctor` answers *can this run at all*. This gate answers *if it runs, will the numbers
mean anything* — and refuses to start a run that cannot produce a like-for-like comparison.

Every test here asserts two things: that the defect is caught, and that the message tells
the operator what to change. A gate that blocks without saying how to proceed is a gate
people learn to bypass.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from safety_eval.config import RunConfig
from safety_eval.reproducibility import Severity, check_config, check_results

ROOT = Path(__file__).resolve().parents[1]


def variant(catalog, tmp_path, *, defaults=None, models=None, tasks=None, drop_routing=False):
    data = yaml.safe_load((ROOT / "config" / "eval_config.yaml").read_text())
    if defaults:
        data["defaults"].update(defaults)
    if models is not None:
        data["models"] = models
    if tasks is not None:
        data["tasks"] = tasks
    if drop_routing:
        data.pop("provider_routing", None)
    path = tmp_path / "v.yaml"
    path.write_text(yaml.safe_dump(data))
    return RunConfig.load(path, catalog)


def issue(verdict, issue_id):
    return next((i for i in verdict.issues if i.id == issue_id), None)


# ------------------------------------------------------------------ the shipped config

def test_the_shipped_configuration_is_reproducible(config) -> None:
    """The gate must pass on what we actually intend to run, or it is theatre."""
    v = check_config(config)
    assert v.reproducible, v.render()
    assert v.exit_code == 0


def test_every_model_is_served_by_one_provider(config) -> None:
    """One engine, one precision, one host — the only way conditions are literally identical."""
    assert len({m.provider for m in config.models}) == 1


# ----------------------------------------------------------------- sample selection

def test_a_cap_without_a_dataset_seed_blocks(catalog, tmp_path) -> None:
    v = check_config(variant(catalog, tmp_path, defaults={"sample_shuffle": None}))
    i = issue(v, "unseeded-selection")
    assert i and i.severity is Severity.BLOCK
    assert "sample_shuffle" in i.correction
    assert "not `seed`" in i.correction, "the fix must distinguish the two seeds"
    assert not v.reproducible


def test_a_cap_below_the_stratum_count_blocks(catalog, tmp_path) -> None:
    v = check_config(variant(catalog, tmp_path, defaults={"limit": 4}))
    i = issue(v, "cap-below-strata")
    assert i and i.severity is Severity.BLOCK
    assert "full dataset" in i.correction


def test_an_unpinned_task_shuffle_blocks(catalog, tmp_path) -> None:
    data = yaml.safe_load((ROOT / "config" / "eval_config.yaml").read_text())
    for task in data["tasks"]:
        if task["key"] == "sycophancy":
            task["args"].pop("shuffle", None)
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(data))
    v = check_config(RunConfig.load(path, catalog))
    i = issue(v, "unpinned-shuffle")
    assert i and i.severity is Severity.BLOCK
    assert i.scope == "sycophancy"


# ------------------------------------------------------------------ provider conditions

def test_unpinned_openrouter_routing_blocks(catalog, tmp_path) -> None:
    """The same model id served at fp8 in one cell and bf16 in another is not one artifact."""
    v = check_config(variant(catalog, tmp_path, models=[
        {"id": "openrouter/meta-llama/llama-3.3-70b-instruct", "family": "meta",
         "label": "Llama 3.3 70B"},
        {"id": "openrouter/qwen/qwen-2.5-72b-instruct", "family": "qwen", "label": "Qwen 72B"},
    ]))
    i = issue(v, "unpinned-routing")
    assert i and i.severity is Severity.BLOCK
    assert "allow_fallbacks: false" in i.correction


def test_mixed_providers_warn(catalog, tmp_path) -> None:
    v = check_config(variant(catalog, tmp_path, models=[
        {"id": "vllm/Qwen/Qwen2.5-7B-Instruct", "family": "qwen", "label": "Qwen"},
        {"id": "openrouter/openai/gpt-4.1", "family": "openai", "label": "GPT-4.1"},
    ]))
    i = issue(v, "mixed-providers")
    assert i and i.severity is Severity.WARN
    assert "separate tables" in i.correction


def test_a_provider_that_discards_a_parameter_blocks(config) -> None:
    """The finding that removed Anthropic from this run.

    Claude Sonnet 4.5 accepts no `seed` on any OpenRouter route. Sending it and reporting it
    as a run condition would state something that never happened to that model.
    """
    support = {config.models[0].id: {"temperature", "max_tokens", "top_p"}}
    v = check_config(config, parameter_support=support)
    i = issue(v, "discarded-parameters")
    assert i and i.severity is Severity.BLOCK
    assert "seed" in i.problem
    assert "symmetric" in i.correction


def test_full_parameter_support_passes(config) -> None:
    support = {m.id: {"temperature", "max_tokens", "seed", "top_p", "top_k"}
               for m in config.models}
    assert check_config(config, parameter_support=support).reproducible


# ------------------------------------------------------- per-benchmark parameter protocol

def test_each_benchmark_declares_its_own_protocol(config, catalog) -> None:
    """Parameters are per benchmark, not per run.

    StrongREJECT is published at temperature 0.75 and XSTest at 0.0. One run-wide number
    would either break StrongREJECT's protocol or make XSTest nondeterministic.
    """
    assert catalog["strong_reject"].generate_params()["temperature"] == 0.75
    assert catalog["xstest"].generate_params()["temperature"] == 0.0
    for bench in catalog:
        assert bench.protocol, f"{bench.key} declares no generation protocol"


def test_protocol_provenance_is_recorded(catalog) -> None:
    """A value the pipeline chose must be distinguishable from the benchmark's own."""
    assert not catalog["xstest"].deviates_from_protocol
    assert not catalog["strong_reject"].deviates_from_protocol
    assert catalog["sycophancy"].deviates_from_protocol, (
        "sycophancy's task sets no GenerateConfig, so our values are pipeline choices"
    )
    for _, _, source, note in catalog["sycophancy"].protocol_rows():
        if "pipeline choice" in source:
            assert note, "a pipeline choice must carry its reasoning into the report"


def test_sampling_above_zero_without_a_seed_blocks(catalog, tmp_path) -> None:
    """StrongREJECT samples at 0.75; without a seed that score cannot be reproduced."""
    v = check_config(variant(catalog, tmp_path, defaults={"seed": None}))
    i = issue(v, "unseeded-sampling")
    assert i and i.severity is Severity.BLOCK
    assert i.scope == "strong_reject"


def test_a_benchmark_with_no_protocol_blocks(catalog, tmp_path) -> None:
    catalog["xstest"].protocol.clear()
    try:
        v = check_config(RunConfig.load(ROOT / "config" / "eval_config.yaml", catalog),
                         catalog)
        i = issue(v, "undeclared-params")
        assert i and i.severity is Severity.BLOCK
        assert "provider defaults" in i.problem
    finally:
        from safety_eval.catalog import Catalog

        fresh = Catalog.load(ROOT / "config" / "benchmarks.yaml")
        catalog.benchmarks["xstest"] = fresh["xstest"]


# ------------------------------------------------------------------------- post-run

def test_divergent_applied_parameters_block(results, config) -> None:
    """Read from evidence, not intent: a provider that ignored a parameter shows up here."""
    for cell in results:
        cell.applied_generate_config = {"temperature": 0.0, "max_tokens": 256}
    results.get("xstest_safe", results.models[1]).applied_generate_config = {
        "temperature": 0.7, "max_tokens": 256}
    v = check_results(results, config)
    assert not v.reproducible
    assert issue(v, "generation-parameters")


def test_divergent_dataset_fingerprint_blocks(results, config) -> None:
    """Evidence that two models were scored on different items."""
    for i, cell in enumerate(c for c in results if c.task_key == "strong_reject"):
        cell.dataset_fingerprint = f"strong_reject_{i}"
    v = check_results(results, config)
    assert issue(v, "dataset-fingerprint")
    assert not v.reproducible


def test_divergent_benchmark_version_blocks(results, config) -> None:
    results.get("xstest_safe", results.models[0]).full_task_version = "3-A"
    assert issue(check_results(results, config), "benchmark-version")


def test_a_parameter_the_harness_did_not_apply_blocks(results, config) -> None:
    """Configured 0.75, harness recorded 0.0 — the recorded value is what made the score."""
    for cell in results:
        if cell.benchmark == "strong_reject":
            cell.applied_generate_config = {"temperature": 0.0, "max_tokens": 2048}
    v = check_results(results, config)
    i = issue(v, "parameter-not-applied")
    assert i and "0.75" in i.problem


def test_a_consistent_run_passes(results, config) -> None:
    for cell in results:
        params = config.catalog[cell.benchmark].generate_params()
        cell.applied_generate_config = dict(params)
        cell.dataset_fingerprint = f"{cell.benchmark}_fixture"
    assert check_results(results, config).reproducible


# ------------------------------------------------------------------------- messaging

def test_every_blocker_states_a_correction(catalog, tmp_path) -> None:
    """A gate that blocks without saying how to proceed is one people learn to bypass."""
    v = check_config(variant(catalog, tmp_path,
                             defaults={"sample_shuffle": None, "limit": 3, "seed": None}))
    assert v.blockers
    for i in v.blockers:
        assert i.correction and len(i.correction) > 30, f"{i.id} has no usable correction"
        assert i.problem, f"{i.id} does not state the problem"


def test_render_says_the_run_was_not_started(catalog, tmp_path) -> None:
    text = check_config(variant(catalog, tmp_path,
                                defaults={"sample_shuffle": None})).render()
    assert "was NOT started" in text
    assert "BLOCKED" in text
