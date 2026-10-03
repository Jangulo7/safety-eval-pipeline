"""Emit an Every Eval Ever record alongside the conditions record.

Why both exist
--------------
Every Eval Ever (schema v0.3.0) is where evaluation results are going to be pooled, so a
record that cannot be expressed in it is a record nobody will read. Emitting EEE is therefore
not a concession; it is the point of interoperating.

But the two formats answer different questions, and the difference is exactly this project's
subject. **EEE v0.3.0 carries no provenance concept at all** -- verified against the installed
schema, which contains no `provenance`, `verified`, `applied` or `measured` field anywhere,
and whose `GenerationArgs` forbids extra keys. So an EEE record states `temperature: 0.75`
without being able to say whether that was read back from the run, resolved by the harness,
typed into a config, or sent to a provider that discarded it.

This module therefore emits both:

* a valid EEE record, so the result can be pooled with everyone else's; and
* the conditions record beside it, carrying the marks EEE has nowhere to put.

Where EEE *can* hold something, it is used rather than duplicated -- the message budget goes
in `eval_limits.message_limit`, which the schema already has. What EEE cannot hold goes in
`additional_details`, which is a free string map, and is clearly labelled as a side channel
rather than passed off as schema support.
"""

from __future__ import annotations

from typing import Any

from .record import ESTABLISHED, Provenance, RunConditionsRecord

SCHEMA_VERSION = "0.3.0"


def to_eee_dict(rec: RunConditionsRecord, *,
                evaluation_name: str,
                score: float,
                metric_name: str,
                source_organization: str,
                lower_is_better: bool,
                evaluator_relationship: str = "first_party",
                source_type: str = "evaluation_run") -> dict[str, Any]:
    """Build an EEE ``EvaluationLog`` payload from a conditions record.

    Returned as a plain dict so that a caller without the EEE package can still produce the
    payload; :func:`validate` turns it into the real model and raises if it does not conform.
    """
    import time

    def value(name: str) -> Any:
        f = rec.get(name)
        return f.value if f else None

    model_id = str(value("model_id") or "unknown")
    library_version = str(value("harness_version") or "unknown")
    timestamp = str(int(time.time()))

    # Fields EEE has a home for.
    generation_args: dict[str, Any] = {
        k: value(k) for k in ("temperature", "top_p", "top_k", "max_tokens")
        if value(k) is not None
    }
    limits = {k: value(k) for k in ("time_limit", "message_limit", "token_limit")
              if value(k) is not None}
    if limits:
        generation_args["eval_limits"] = limits

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": f"{evaluation_name}/{model_id}/{timestamp}",
        "retrieved_timestamp": timestamp,
        "source_metadata": {
            "source_type": source_type,
            "source_organization_name": source_organization,
            "evaluator_relationship": evaluator_relationship,
            # The side channel, and labelled as one. These are conditions the schema has no
            # field for; putting them here keeps the record valid without pretending EEE
            # supports them.
            "additional_details": _provenance_sidecar(rec),
        },
        "eval_library": {
            "name": str(value("harness_name") or "unknown"),
            "version": library_version,
        },
        "model_info": {"name": model_id, "id": model_id},
        "evaluation_results": [{
            "evaluation_name": evaluation_name,
            "source_data": _source_data(rec),
            # `lower_is_better` is required and has no default, which is right: a score
            # whose direction is unstated cannot be compared or aggregated, and guessing it
            # from the metric's name would be the kind of inference this project objects to.
            "metric_config": {"metric_name": metric_name,
                              "lower_is_better": lower_is_better},
            "score_details": {"score": score},
            "generation_config": {
                "generation_args": generation_args or None,
                "additional_details": _conditions_sidecar(rec),
            },
        }],
    }


def _source_data(rec: RunConditionsRecord) -> dict[str, Any]:
    """Which dataset, in whichever of EEE's three source shapes fits what is known."""
    field = rec.get("dataset_name")
    name = str(field.value) if field and field.value else None
    if name and "/" in name:
        return {"source_type": "hf_dataset", "dataset_name": name, "hf_repo": name}
    # `SourceDataPrivate` has only a name and a free string map, so the reason the dataset
    # could not be identified goes in the map. Stating it matters: a consumer that sees only
    # "unidentified" cannot tell a private dataset from a log that failed to record one.
    return {"source_type": "other",
            "dataset_name": name or "unidentified",
            "additional_details": {
                "rcr_reason": ("the source log identifies no dataset" if not name else
                               "dataset named but not resolvable to a public repository")}}


def _provenance_sidecar(rec: RunConditionsRecord) -> dict[str, str]:
    """The marks EEE has nowhere to put, as strings, plus what they add up to.

    Carried so that a pooled record does not silently lose the distinction between a condition
    that was established and one that was merely requested. A reader who ignores this map gets
    a perfectly ordinary EEE record; a reader who reads it can tell the two apart.
    """
    counts = rec.counts()
    out = {
        "rcr_tier": rec.tier.value,
        "rcr_tier_note": "unvalidated comparability tier; see rcr-core",
        "rcr_missing_for_next_tier": ", ".join(rec.missing_for_next_tier) or "none",
        "rcr_provenance_counts": ", ".join(f"{k}={v}" for k, v in counts.items()),
        "rcr_note": ("EEE v0.3.0 has no provenance field, so these marks travel here. "
                     "Fields marked requested were sent somewhere and never confirmed."),
    }
    established = sorted(n for n, f in rec.fields.items() if f.established)
    requested = sorted(n for n, f in rec.fields.items()
                       if f.provenance is Provenance.REQUESTED)
    out["rcr_established_fields"] = ", ".join(established) or "none"
    out["rcr_requested_unverified_fields"] = ", ".join(requested) or "none"
    return out


def _conditions_sidecar(rec: RunConditionsRecord) -> dict[str, str] | None:
    """Per-field marks for the conditions EEE does carry, plus the serving facts it does not.

    `additional_details` is a flat string map, so each entry is rendered as
    ``value (provenance)`` -- losing structure but not the mark, which is the part that cannot
    be reconstructed later.
    """
    out: dict[str, str] = {}
    for name in ("temperature", "top_p", "max_tokens", "seed", "epochs",
                 "message_limit", "dataset_fingerprint", "served_provider",
                 "served_provider_mix", "quant_scheme", "samples_scored",
                 "samples_unscored"):
        f = rec.get(name)
        if f is None:
            continue
        if f.value is None and f.provenance not in ESTABLISHED:
            # Record the absence, with the reason. An EEE consumer that sees nothing cannot
            # tell "not set" from "set and unrecoverable", and those are different facts.
            out[name] = f"unavailable ({f.note})" if f.note else "unavailable"
        else:
            out[name] = f"{f.value} ({f.provenance.value})"
    return out or None


def validate(payload: dict[str, Any]) -> Any:
    """Validate against the installed Every Eval Ever schema. Raises if it does not conform.

    Uses EEE's own model rather than a copy of the schema, so this test fails if the upstream
    schema moves -- which is the point of claiming compatibility with it.
    """
    from every_eval_ever.eval_types import EvaluationLog

    return EvaluationLog.model_validate(payload)
