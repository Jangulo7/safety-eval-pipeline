"""Map an AISI Inspect evaluation log onto a Run Conditions Record.

Reads only the log. It does not import the pipeline that lives beside it, does not require
the log to have been produced by that pipeline, and does not assume any field is present --
an adapter that only works on its author's own logs cannot support a claim about anyone
else's.

The central distinction this adapter draws
------------------------------------------
Inspect records generation parameters in two places that are easy to confuse:

* ``EvalSpec.model_generate_config`` -- what the harness *resolved and applied*;
* the per-sample model events -- what the provider *answered with*.

A value present in the first is marked APPLIED. A value the record shows was sent but that
nothing confirms took effect is REQUESTED. The serving provider and the quantization kernel
are only ever MEASURED, because they can only come from the response or the server's own
report. Where Inspect has nothing to say, the field is UNAVAILABLE with the reason, never
back-filled from the request.
"""

from __future__ import annotations

from typing import Any

from .record import Provenance, RunConditionsRecord

EMITTER = "rcr-inspect/0.1.0"

# Generation parameters worth recording. Inspect resolves these into the eval spec, so their
# presence there is evidence that the harness applied them -- not proof the provider honoured
# them, which is a separate question no log can settle on its own.
GENERATION_FIELDS = ("temperature", "max_tokens", "top_p", "top_k", "seed",
                     "frequency_penalty", "presence_penalty", "num_choices")

# Budgets. Absent from most conditions reporting and load-bearing for agentic evaluation: a
# task that stopped because it ran out of steps did not refuse.
LIMIT_FIELDS = ("message_limit", "token_limit", "time_limit", "working_limit")


def _get(obj: Any, *names: str) -> Any:
    """First present attribute, tolerating the shape differences between Inspect versions."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def from_eval_log(log: Any, source: str = "") -> RunConditionsRecord:
    """Emit a record from an already-loaded Inspect ``EvalLog``."""
    rec = RunConditionsRecord(source=source or str(_get(log, "location") or ""),
                              emitter=EMITTER)
    spec = _get(log, "eval")

    # --- identity -----------------------------------------------------------------------
    model = _get(spec, "model")
    rec.add("model_id", model, Provenance.MEASURED if model else Provenance.UNAVAILABLE,
            "" if model else "the log names no model")

    # The provider prefix is the router or runtime asked, not the upstream that answered.
    if isinstance(model, str) and "/" in model:
        rec.add("model_provider_requested", model.split("/", 1)[0], Provenance.REQUESTED,
                "the prefix of the model identifier; which upstream answered is a "
                "separate field")

    task = _get(spec, "task")
    rec.add("task", task, Provenance.MEASURED if task else Provenance.UNAVAILABLE)
    version = _get(spec, "task_version")
    rec.add("task_version", version,
            Provenance.MEASURED if version is not None else Provenance.UNAVAILABLE)

    # --- harness ------------------------------------------------------------------------
    pkg = _get(spec, "packages") or {}
    harness_version = None
    if isinstance(pkg, dict):
        harness_version = pkg.get("inspect_ai") or next(iter(pkg.values()), None)
    rec.add("harness_name", "inspect_ai", Provenance.MEASURED)
    rec.add("harness_version", harness_version,
            Provenance.MEASURED if harness_version else Provenance.UNAVAILABLE,
            "" if harness_version else "the log records no package versions")

    revision = _get(spec, "revision")
    if revision is not None:
        rec.add("code_revision", _get(revision, "commit"), Provenance.MEASURED)
        dirty = getattr(revision, "dirty", None)
        if dirty is not None:
            rec.add("code_dirty", bool(dirty), Provenance.MEASURED,
                    "a dirty tree means the commit does not describe what ran")

    # --- dataset ------------------------------------------------------------------------
    dataset = _get(spec, "dataset")
    name = _get(dataset, "name", "location") if dataset is not None else None
    rec.add("dataset_name", name,
            Provenance.MEASURED if name else Provenance.UNAVAILABLE,
            "" if name else "the log identifies no dataset")
    fingerprint = _get(dataset, "sha256", "fingerprint") if dataset is not None else None
    note = "recorded by the harness"
    if not fingerprint and dataset is not None:
        # Inspect does not record a content hash for the dataset, but it does record the
        # per-sample ids, and those ids are themselves content-derived. Hashing the sorted
        # set reconstructs a fingerprint that answers the question the field exists for: did
        # two runs score the same items? Derived from the log's own contents, so it is
        # measured rather than asserted -- and the derivation is named, because a fingerprint
        # whose construction is unstated cannot be compared with anyone else's.
        fingerprint = _fingerprint_from_sample_ids(_get(dataset, "sample_ids"))
        note = ("derived: sha256 of the sorted sample ids, which are content-derived. The "
                "harness records no dataset hash of its own")
    rec.add("dataset_fingerprint", fingerprint,
            Provenance.MEASURED if fingerprint else Provenance.UNAVAILABLE,
            note if fingerprint else
            "the log records neither a dataset hash nor sample ids, so two runs cannot be "
            "shown to have scored the same items")
    shuffled = _get(dataset, "shuffled") if dataset is not None else None
    if shuffled is not None:
        rec.add("dataset_shuffled", bool(shuffled), Provenance.MEASURED,
                "item order affects which items a capped run reaches")
    total = _get(dataset, "samples") if dataset is not None else None
    if total is not None:
        rec.add("dataset_samples_total", total, Provenance.MEASURED)

    # --- what the harness applied -------------------------------------------------------
    applied = _get(spec, "model_generate_config")
    dumped = applied.model_dump() if hasattr(applied, "model_dump") else {}
    for key in GENERATION_FIELDS:
        value = dumped.get(key)
        if value is None:
            rec.add(key, None, Provenance.UNAVAILABLE,
                    "not recorded as applied; either it was never set, or it was sent and "
                    "the harness did not read it back")
        else:
            rec.add(key, value, Provenance.APPLIED,
                    "resolved by the harness; whether the provider honoured it is a "
                    "separate question this log cannot answer")

    config = _get(spec, "config")
    cdump = config.model_dump() if hasattr(config, "model_dump") else {}
    for key in ("epochs", "limit", "sample_shuffle", "fail_on_error", "max_connections"):
        if cdump.get(key) is not None:
            rec.add(key, cdump[key], Provenance.APPLIED)
    for key in LIMIT_FIELDS:
        value = cdump.get(key)
        rec.add(key, value, Provenance.APPLIED if value is not None else Provenance.UNAVAILABLE,
                "" if value is not None else
                "no budget recorded; a run that stopped at a limit cannot be told from one "
                "that finished")

    # --- what actually served -----------------------------------------------------------
    _add_serving(rec, log)

    # --- outcome shape ------------------------------------------------------------------
    results = _get(log, "results")
    scored = _get(results, "completed_samples") if results is not None else None
    total_s = _get(results, "total_samples") if results is not None else None
    rec.add("samples_scored", scored,
            Provenance.MEASURED if scored is not None else Provenance.UNAVAILABLE)
    if total_s is not None and scored is not None:
        rec.add("samples_unscored", int(total_s) - int(scored), Provenance.MEASURED,
                "samples that left the denominator; a rate that moved because scoring failed "
                "is not a finding about the model")

    status = _get(log, "status")
    rec.add("status", status, Provenance.MEASURED if status else Provenance.UNAVAILABLE)
    return rec


def _fingerprint_from_sample_ids(sample_ids: Any) -> str | None:
    """A stable fingerprint of the item set, from ids the harness already content-derives.

    Sorted before hashing so that two runs over the same items agree regardless of the order
    they were served in -- order is recorded separately, because it is a different condition.
    """
    import hashlib

    if not sample_ids:
        return None
    joined = "\n".join(sorted(str(s) for s in sample_ids))
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _add_serving(rec: RunConditionsRecord, log: Any) -> None:
    """Which upstream answered, counted across samples rather than taken from the first.

    Routing happens per request. A single evaluation can be spread across several upstreams
    that differ in weights, kernel, precision and chat template, and its score is then a
    property of no one system -- so the mix is recorded, not just the majority.

    Only the model under test counts. A judged benchmark also calls a grader, and if that
    grader goes through the same router its response carries a provider too; counting it
    would report the grader's upstream as having served the evaluation.
    """
    subject = str(_get(_get(log, "eval"), "model") or "")
    mix: dict[str, int] = {}
    for sample in (_get(log, "samples") or []):
        for event in (getattr(sample, "events", None) or []):
            if subject and str(getattr(event, "model", "")) != subject:
                continue
            call = getattr(event, "call", None)
            response = getattr(call, "response", None) if call else None
            if isinstance(response, dict) and response.get("provider"):
                provider = str(response["provider"])
                mix[provider] = mix.get(provider, 0) + 1

    if not mix:
        rec.add("served_provider", None, Provenance.UNAVAILABLE,
                "no serving provider in any response; for a local runtime this is expected, "
                "and for a routed one it means the upstream that produced these scores "
                "cannot be established from the log")
        rec.add("quant_scheme", None, Provenance.UNAVAILABLE,
                "numerical precision is not recorded in the log; it can only come from the "
                "serving runtime's own report")
        return

    ordered = dict(sorted(mix.items(), key=lambda kv: -kv[1]))
    rec.add("served_provider", next(iter(ordered)), Provenance.MEASURED,
            "read from the router's own response")
    rec.add("served_provider_mix", ordered, Provenance.MEASURED,
            "requests per upstream" + ("" if len(ordered) == 1 else
                                       "; more than one upstream served this single run, so "
                                       "its score averages over different serving stacks"))
    rec.add("quant_scheme", None, Provenance.UNAVAILABLE,
            "the router reports which upstream answered but not the numerical precision it "
            "served at; a declared precision, where one exists, is the provider's claim "
            "rather than a measurement")


def from_log_file(path: str) -> RunConditionsRecord:
    """Read an Inspect ``.eval`` file and emit its record."""
    from inspect_ai.log import read_eval_log

    return from_eval_log(read_eval_log(path), source=str(path))
