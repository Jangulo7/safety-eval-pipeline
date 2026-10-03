"""The Run Conditions Record library.

Tested against hand-built logs rather than fixtures from the pipeline beside it, because the
claim the library makes is that it works on logs it did not produce. A test suite that only
feeds it this project's own output would prove the opposite of what is wanted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from rcr_core import Provenance, RunConditionsRecord, Tier
from rcr_core.inspect_adapter import _fingerprint_from_sample_ids, from_eval_log

# --------------------------------------------------------------------------- the record

def test_only_measured_and_applied_count_as_established() -> None:
    """The distinction the whole record exists to draw.

    `requested` carries a value and establishes nothing: a provider that discarded it leaves
    the field looking exactly as it would had it been honoured.
    """
    rec = RunConditionsRecord()
    rec.add("a", 1, Provenance.MEASURED)
    rec.add("b", 2, Provenance.APPLIED)
    rec.add("c", 3, Provenance.REQUESTED)
    rec.add("d", 4, Provenance.EDITORIAL)
    assert rec.established("a") and rec.established("b")
    assert not rec.established("c"), "a request is not a condition"
    assert not rec.established("d")


def test_a_field_marked_measured_but_empty_is_not_established() -> None:
    """A mark is a claim about how a value was obtained, not a substitute for having one."""
    rec = RunConditionsRecord()
    rec.add("x", None, Provenance.MEASURED)
    assert not rec.established("x")


def test_tiers_are_cumulative() -> None:
    """Reading back the serving precision does not rescue a record that cannot say which
    items were scored: two runs over different items are not comparable however well their
    hardware is described."""
    rec = RunConditionsRecord()
    for name in ("model_id", "harness_name", "harness_version", "dataset_name"):
        rec.add(name, "v", Provenance.MEASURED)
    for name in ("served_provider", "quant_scheme"):
        rec.add(name, "v", Provenance.MEASURED)
    assert rec.tier is Tier.T1, "tier 3 fields cannot skip tier 2"
    assert "dataset_fingerprint" in rec.missing_for_next_tier


def test_a_record_that_establishes_nothing_is_tier_zero() -> None:
    rec = RunConditionsRecord()
    rec.add("model_id", "m", Provenance.REQUESTED)
    assert rec.tier is Tier.T0


def test_the_tier_names_exactly_what_would_improve_it() -> None:
    """A tier without an account of what it costs to improve is a grade, not a diagnosis."""
    rec = RunConditionsRecord()
    for name in ("model_id", "harness_name", "harness_version", "dataset_name",
                 "temperature", "max_tokens", "epochs", "dataset_fingerprint",
                 "samples_scored"):
        rec.add(name, 1, Provenance.MEASURED)
    assert rec.tier is Tier.T2
    assert sorted(rec.missing_for_next_tier) == ["quant_scheme", "served_provider"]


def test_a_full_record_reaches_the_top_tier_and_asks_for_nothing() -> None:
    rec = RunConditionsRecord()
    for name in ("model_id", "harness_name", "harness_version", "dataset_name",
                 "temperature", "max_tokens", "epochs", "dataset_fingerprint",
                 "samples_scored", "served_provider", "quant_scheme"):
        rec.add(name, 1, Provenance.MEASURED)
    assert rec.tier is Tier.T3
    assert rec.missing_for_next_tier == []


def test_the_serialised_record_carries_the_marks_and_the_tier() -> None:
    rec = RunConditionsRecord(source="x.eval", emitter="t/1")
    rec.add("model_id", "m", Provenance.MEASURED, "from the log")
    out = json.loads(rec.to_json())
    assert out["fields"]["model_id"] == {"value": "m", "provenance": "measured",
                                         "note": "from the log"}
    assert out["tier"] == "0" and "provenance_counts" in out


# ------------------------------------------------------------------- the inspect adapter

def _log(**over):
    """A minimal stand-in for an Inspect EvalLog, built by hand."""
    spec = SimpleNamespace(
        model=over.get("model", "openrouter/meta-llama/llama-3.1-8b-instruct"),
        task="demo", task_version=1,
        packages={"inspect_ai": "0.3.260"},
        revision=SimpleNamespace(commit="abc123", dirty=False),
        dataset=SimpleNamespace(name="demo/set", location="demo/set",
                                samples=2, shuffled=True,
                                sample_ids=over.get("sample_ids", ["s_b", "s_a"])),
        model_generate_config=SimpleNamespace(
            model_dump=lambda: over.get("gen", {"temperature": 0.0, "max_tokens": 64})),
        config=SimpleNamespace(
            model_dump=lambda: over.get("cfg", {"epochs": 1, "message_limit": 20})),
    )
    return SimpleNamespace(eval=spec, status="success", location="demo.eval",
                           results=SimpleNamespace(completed_samples=2, total_samples=2),
                           samples=over.get("samples", []))


def test_applied_parameters_are_not_promoted_to_measured() -> None:
    """Inspect resolves generation parameters, which shows the harness applied them. It does
    not show the provider honoured them, and the record must not claim that it does."""
    rec = from_eval_log(_log())
    assert rec.get("temperature").provenance is Provenance.APPLIED
    assert rec.get("max_tokens").provenance is Provenance.APPLIED


def test_an_unset_parameter_is_unavailable_with_a_reason_not_a_default() -> None:
    rec = from_eval_log(_log(gen={"temperature": 0.0}))
    top_p = rec.get("top_p")
    assert top_p.provenance is Provenance.UNAVAILABLE
    assert top_p.value is None
    assert top_p.note, "an unavailable field must say why"


def test_the_model_prefix_is_a_request_not_the_upstream_that_answered() -> None:
    rec = from_eval_log(_log())
    assert rec.get("model_provider_requested").provenance is Provenance.REQUESTED


def test_the_message_budget_is_recorded_when_present() -> None:
    """Absent from most conditions reporting and load-bearing: a task that stopped because it
    ran out of steps did not refuse."""
    rec = from_eval_log(_log())
    assert rec.get("message_limit").value == 20
    assert rec.get("message_limit").provenance is Provenance.APPLIED


def test_a_missing_budget_says_what_cannot_be_distinguished() -> None:
    rec = from_eval_log(_log(cfg={"epochs": 1}))
    f = rec.get("message_limit")
    assert f.provenance is Provenance.UNAVAILABLE
    assert "stopped at a limit" in f.note


# ------------------------------------------------------------------------- fingerprinting

def test_the_fingerprint_is_order_independent() -> None:
    """Two runs that scored the same items agree, whatever order they were served in -- order
    is a separate condition and is recorded separately."""
    a = _fingerprint_from_sample_ids(["s_a", "s_b", "s_c"])
    b = _fingerprint_from_sample_ids(["s_c", "s_a", "s_b"])
    assert a == b and a.startswith("sha256:")


def test_a_different_item_set_gives_a_different_fingerprint() -> None:
    assert _fingerprint_from_sample_ids(["s_a"]) != _fingerprint_from_sample_ids(["s_b"])


def test_the_fingerprint_is_derived_and_says_so() -> None:
    """The harness records no dataset hash. One is reconstructed from the content-derived
    sample ids, and the derivation is named: a fingerprint whose construction is unstated
    cannot be compared with anyone else's."""
    rec = from_eval_log(_log())
    f = rec.get("dataset_fingerprint")
    assert f.provenance is Provenance.MEASURED
    assert "derived" in f.note and "sorted sample ids" in f.note


def test_no_sample_ids_means_no_fingerprint_rather_than_a_hash_of_nothing() -> None:
    rec = from_eval_log(_log(sample_ids=[]))
    f = rec.get("dataset_fingerprint")
    assert f.value is None and f.provenance is Provenance.UNAVAILABLE


# ----------------------------------------------------------------------------- serving

def _event(model, provider):
    return SimpleNamespace(model=model,
                           call=SimpleNamespace(response={"provider": provider}))


def test_the_serving_provider_is_counted_across_samples() -> None:
    """Routing happens per request, so one evaluation can be split across upstreams that
    differ in weights, kernel and chat template. The mix is the finding, not the majority."""
    subject = "openrouter/meta-llama/llama-3.1-8b-instruct"
    samples = ([SimpleNamespace(events=[_event(subject, "Novita")])] * 3
               + [SimpleNamespace(events=[_event(subject, "DeepInfra")])])
    rec = from_eval_log(_log(samples=samples))
    assert rec.get("served_provider").value == "Novita"
    assert rec.get("served_provider_mix").value == {"Novita": 3, "DeepInfra": 1}
    assert "more than one upstream" in rec.get("served_provider_mix").note


def test_the_grader_is_not_counted_as_serving_the_evaluation() -> None:
    """A judged benchmark calls two models per sample. If the grader goes through the same
    router its response carries a provider too, and counting it reports the grader's upstream
    as having served the evaluation."""
    subject = "openrouter/meta-llama/llama-3.1-8b-instruct"
    samples = [SimpleNamespace(events=[_event(subject, "Cloudflare"),
                                       _event("openrouter/openai/gpt-4.1-mini", "OpenAI")])]
    rec = from_eval_log(_log(samples=samples))
    assert rec.get("served_provider_mix").value == {"Cloudflare": 1}


def test_a_log_with_no_serving_evidence_says_so_rather_than_guessing() -> None:
    rec = from_eval_log(_log(samples=[]))
    assert rec.get("served_provider").provenance is Provenance.UNAVAILABLE
    assert rec.get("quant_scheme").provenance is Provenance.UNAVAILABLE
    assert rec.get("served_provider").note


def test_precision_is_never_inferred_from_the_request() -> None:
    """The router says which upstream answered, not at what numerical precision. A declared
    precision, where one exists, is the provider's claim rather than a measurement."""
    subject = "openrouter/meta-llama/llama-3.1-8b-instruct"
    rec = from_eval_log(_log(samples=[SimpleNamespace(events=[_event(subject, "Novita")])]))
    assert rec.get("quant_scheme").provenance is Provenance.UNAVAILABLE


# ------------------------------------------------------------------------ foreign logs

def test_the_adapter_survives_a_log_that_is_missing_almost_everything() -> None:
    """Logs from elsewhere will not have this project's conventions. Absence must produce an
    unavailable field with a reason, never an exception and never a fabricated default."""
    bare = SimpleNamespace(eval=SimpleNamespace(), status=None, location="", results=None,
                           samples=None)
    rec = from_eval_log(bare)
    assert rec.tier is Tier.T0
    assert rec.get("model_id").provenance is Provenance.UNAVAILABLE
    assert rec.get("dataset_fingerprint").provenance is Provenance.UNAVAILABLE
    json.loads(rec.to_json())          # still serialises


def test_the_adapter_does_not_import_the_pipeline_beside_it() -> None:
    """The library's claim is that it reads any Inspect log. Coupling it to this repository's
    own result types would make that claim untestable."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "rcr_core"
    for py in src.glob("*.py"):
        text = py.read_text()
        assert "safety_eval" not in text, f"{py.name} imports the pipeline"
