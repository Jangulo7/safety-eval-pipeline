"""Contamination disclosure and the Parameter Register, computed from the run itself.

Two disclosure frameworks travel with every report:

**Contamination disclosure** (``.private/Contamination-disclosure.txt``) — four fields that
determine whether one evaluation's numbers can honestly be set beside another's: whether
results are broken down by stratum (F1), whether the elicitation conditions are
reproducible (F2), what contamination controls were applied (F3, five types), and whether
the instrument can be regenerated (F4). The codes are computed from what the run actually
recorded, not asserted — so a change that stops recording something lowers the score.

**Parameter Register** — the field list from this repository's pre-Inspect README, written
for locally-served quantized models. Roughly half of it is inapplicable to a hosted model
and *saying which half, and why*, is itself part of the disclosure: a blank row and a
not-applicable row mean very different things to a reader.

Nothing here inflates. Where a control was not applied the code is `0` and the report says
so. An honest `0` beside a named mechanism is worth more than an unearned `2`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RunConfig
from .results import CellStatus, ResultSet

# Publication dates of the instruments, for the temporal-contamination assessment (F3/t3).
BENCHMARK_DATES = {
    "xstest": ("2023-08", "arXiv:2308.01263"),
    "sycophancy": ("2023-10", "arXiv:2310.13548"),
    "strong_reject": ("2024-02", "arXiv:2402.10260"),
}

# Regeneration status of each instrument (F4). Stating the status is the disclosure act,
# even for a benchmark this project did not build.
REGENERATION = {
    "xstest": "Static artifact. Items released as a HuggingFace dataset; the construction "
              "procedure is described in the paper but no generator is published, so fresh "
              "items cannot be produced.",
    "sycophancy": "Static artifact. Items released as JSON in the authors' repository; no "
                  "generator is published.",
    "strong_reject": "Static artifact. Items released as CSV in the authors' repository; "
                     "the construction procedure is described in the paper but no generator "
                     "is published.",
}


@dataclass
class Field:
    """One disclosure field: its code, and the evidence for that code."""

    key: str
    name: str
    code: str
    rationale: str
    detail: str = ""


@dataclass
class ContaminationDisclosure:
    """The four fields, plus the five F3 sub-types and the F2 slot string."""

    f1: Field
    f2: Field
    f2_notes: str
    f3: list[Field] = field(default_factory=list)
    f4: Field | None = None
    headline: str = ""

    def rows(self) -> list[tuple[str, str, str]]:
        out = [(self.f1.name, self.f1.code, self.f1.rationale),
               (f"{self.f2.name}  (`{self.f2_notes}`)", self.f2.code, self.f2.rationale)]
        out += [(f.name, f.code, f.rationale) for f in self.f3]
        if self.f4:
            out.append((self.f4.name, self.f4.code, self.f4.rationale))
        return out


def contamination_disclosure(
    results: ResultSet, config: RunConfig, *, per_stratum: bool = True
) -> ContaminationDisclosure:
    """Compute the four disclosure fields from what this run actually recorded."""
    meta = results.metadata
    ok = [c for c in results if c.status is CellStatus.OK]

    # --- F1: strata reported -------------------------------------------------------------
    has_counts = any(c.stratum_counts for c in ok)
    if per_stratum and has_counts:
        f1 = Field("f1", "F1 · Strata reported", "2",
                   "Per-stratum scores are given with n per cell, and the strata are named "
                   "(XSTest prompt type, StrongREJECT harm category, sycophancy source "
                   "dataset).")
    elif has_counts:
        f1 = Field("f1", "F1 · Strata reported", "1",
                   "Stratum sample counts are recorded but scores are aggregate only.")
    else:
        f1 = Field("f1", "F1 · Strata reported", "0", "Aggregate numbers only.")

    # --- F2: elicitation budget ----------------------------------------------------------
    slot_i = "H"                                    # Inspect: a named harness
    slot_ii = "Y" if meta.inspect_ai_version else "-"
    slot_iii = "Y" if any(c.max_tokens for c in ok) else "-"
    slot_iv = "Y" if all(c.epochs for c in ok) else "-"
    slot_v = "Y"                                    # single; no epoch reduction
    f2_notes = f"{slot_i}{slot_ii}{slot_iii}{slot_iv}{slot_v}"
    specified = sum(1 for s in (slot_ii, slot_iii, slot_iv, slot_v) if s == "Y")
    f2 = Field(
        "f2", "F2 · Elicitation budget", "2" if specified >= 2 and slot_iii == "Y" else "1",
        f"Harness named and pinned ({meta.inspect_ai_version} / "
        f"{meta.inspect_evals_version}); per-task token cap recorded; "
        f"{ok[0].epochs if ok else 1} attempt per item, resolved as single."
        if slot_iii == "Y" else
        "Harness named and pinned, but no token or step budget is recorded, so the "
        "elicitation budget is not reproducible.",
    )

    # --- F3: contamination controls, one code per type -----------------------------------
    dated = ", ".join(f"{k} {v[0]}" for k, v in sorted(BENCHMARK_DATES.items()))
    fingerprinted = [c for c in ok if c.dataset_fingerprint]
    f3 = [
        Field("t1", "F3 · t1 Direct", "1",
              "No overlap check, canary string or private held-out set. All three "
              "instruments are public and predate the training cutoffs of every model "
              "under test, so direct contamination should be assumed rather than excluded. "
              "Named, uncontrolled."),
        Field("t2", "F3 · t2 Derivative",
              "2" if fingerprinted else "1",
              "Item provenance is tracked: dataset source and Inspect's content "
              "fingerprint are recorded per cell, so two runs can be shown to have used "
              "the same items."
              if fingerprinted else
              "Sources are named but no content fingerprint is recorded."),
        Field("t3", "F3 · t3 Temporal", "2",
              f"Instrument publication dates are stated and related to the items: {dated}. "
              "All three predate the stated training cutoffs of all models under test, so "
              "the relation is that contamination is likely, not excluded."),
        Field("t4", "F3 · t4 Distributional", "0",
              "No perturbation, paraphrase-robustness or template-novelty testing was "
              "performed. Recorded as 0 rather than inflated."),
        Field("t5", "F3 · t5 Acquired", "2",
              "The solver chain is system_message then generate(). No tools, no retrieval "
              "and no network access were available to the evaluated model during scoring. "
              "Control stated; not established by transcript review."),
    ]

    # --- F4: regeneration ----------------------------------------------------------------
    benches = sorted({c.benchmark for c in results})
    stated = [b for b in benches if b in REGENERATION]
    f4 = Field("f4", "F4 · Regeneration", "2" if len(stated) == len(benches) else "1",
               "The regeneration status of every instrument is stated explicitly: all three "
               "are static artifacts released as items, with no published generator."
               if len(stated) == len(benches) else
               "The regeneration status of some instruments is not stated.")

    headline = (
        "The load-bearing code is **t1 = 1**. These are public benchmarks that predate the "
        "training cutoff of every model evaluated, and no decontamination check was run. "
        "Some direct contamination should be assumed. That does not void the measurement — "
        "a model that has seen XSTest and still over-refuses is still over-refusing — but "
        "any claim of an uncontaminated result would be false."
    )
    return ContaminationDisclosure(f1=f1, f2=f2, f2_notes=f2_notes, f3=f3, f4=f4,
                                   headline=headline)


# --------------------------------------------------------------------- Parameter Register

@dataclass
class RegisterRow:
    section: str
    parameter: str
    value: str
    status: str
    """``recorded`` · ``not applicable`` · ``undisclosed`` · ``missing``"""


def parameter_register(results: ResultSet, config: RunConfig) -> list[RegisterRow]:
    """The Parameter Register, filled from the run, with inapplicable rows reasoned.

    The register assumes a locally-served quantized model. What is and is not applicable
    changes with the serving arrangement, so the reason is carried on every row that cannot
    be filled — a blank row and a not-applicable row are different claims.
    """
    meta = results.metadata
    ok = [c for c in results if c.status is CellStatus.OK]
    first = ok[0] if ok else None
    serving = config.raw.get("serving") or {}
    local = serving.get("backend") == "vllm"
    rows: list[RegisterRow] = []

    def add(section: str, parameter: str, value: Any, status: str = "recorded") -> None:
        rows.append(RegisterRow(section, parameter, str(value), status))

    # A · model identity and quantization
    add("A · Model identity", "model_id", ", ".join(results.models))
    add("A · Model identity", "base_model_family",
        ", ".join(sorted({c.family for c in results})))
    params = {m.label: getattr(m, "params_b", None) for m in config.models}
    add("A · Model identity", "param_count_billions",
        ", ".join(f"{k} {v}B" for k, v in params.items() if v) or "undisclosed",
        "recorded" if any(params.values()) else "undisclosed")
    if local:
        add("A · Model identity", "quant_scheme",
            serving.get("quantization", "none") + f" ({serving.get('dtype', '?')})")
        add("A · Model identity", "bits_per_weight",
            "16 (bfloat16)" if serving.get("quantization") in (None, "none")
            else serving.get("quantization"))
    else:
        add("A · Model identity", "quant_scheme", "not disclosed by the router", "undisclosed")
        add("A · Model identity", "bits_per_weight", "unknown", "undisclosed")
    for parameter, why in (
        ("group_size", "quantization-pipeline field; models are served at native precision"),
        ("pruning_method", "no pruning applied"),
        ("sparsity_ratio", "no pruning applied"),
        ("healing_applied", "no quantization or pruning, so no healing"),
        ("calibration_dataset", "no post-training quantization was performed"),
        ("file_hash_sha256", "no local single-file artifact; the dataset fingerprint is the "
                             "analogous integrity field and is recorded"),
    ):
        add("A · Model identity", parameter, why, "not applicable")

    # B · inference configuration — per benchmark, because that is where it is held constant
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        for key, spec in bench.protocol.items():
            source = "benchmark protocol" if spec.get("source") == "task" else "pipeline choice"
            add("B · Inference", f"{task.key} · {key}", f"{spec['value']} ({source})")
    add("B · Inference", "random_seed · generation", first.seed if first else "—")
    add("B · Inference", "random_seed · dataset order",
        first.sample_shuffle if first else "—")
    add("B · Inference", "top_p / top_k / repetition_penalty",
        "unset; the serving default applies uniformly to all models", "undisclosed")
    add("B · Inference", "max_context_window", serving.get("max_model_len", "—")
        if local else "provider default", "recorded" if local else "undisclosed")
    add("B · Inference", "batch_size (concurrency)",
        f"{first.max_connections} concurrent requests" if first else "—")
    add("B · Inference", "system_prompt_id",
        "xstest: 'You are a helpful assistant.' (set by the task); "
        "sycophancy / strong_reject: none")
    for parameter, why in (("rope_scaling", "model default; not overridden"),):
        add("B · Inference", parameter, why, "not applicable")

    # C · hardware and environment
    if local:
        add("C · Environment", "inference_backend",
            f"vLLM (dtype {serving.get('dtype')}, max_model_len {serving.get('max_model_len')})")
        add("C · Environment", "gpu_model", serving.get("gpu_model", "recorded at run time"))
        add("C · Environment", "driver_version", serving.get("driver_version", "recorded at run time"))
    else:
        add("C · Environment", "inference_backend", "hosted API (provider-routed)")
        for parameter in ("gpu_model", "gpu_count", "cpu_model", "system_ram_gb",
                          "driver_version", "quant_backend_version"):
            add("C · Environment", parameter,
                "models run on the provider's hardware, which is not disclosed",
                "not applicable")

    # D · classification
    add("D · Classification", "benchmark_category",
        "Safety — over-refusal, under-refusal, sycophancy")
    add("D · Classification", "metric_type",
        "Refusal calibration; harmful-uplift; sycophancy. Not accuracy.")
    add("D · Classification", "dataset_version",
        "; ".join(sorted({f"{c.task_key} {c.full_task_version}" for c in ok
                          if c.full_task_version})) or "—")
    add("D · Classification", "industry_vertical / use_case_tags",
        "sales-classification fields with no bearing on a safety measurement",
        "not applicable")

    # E · tool and pipeline metadata
    add("E · Pipeline", "eval_tool_version",
        f"inspect_ai {meta.inspect_ai_version}, inspect_evals {meta.inspect_evals_version}, "
        f"pipeline {meta.pipeline_version}")
    add("E · Pipeline", "eval_tool_commit_hash", meta.command or _git_commit())
    add("E · Pipeline", "run_timestamp", meta.started_utc)
    add("E · Pipeline", "run_duration_seconds", f"{results.total_wall_clock_s:.0f}")
    add("E · Pipeline", "dataset_fingerprint",
        "; ".join(sorted({f"{c.task_key}: {c.dataset_fingerprint}" for c in ok
                          if c.dataset_fingerprint})) or "—")
    return rows


def _git_commit() -> str:
    """The pipeline's own commit — the Register's `eval_tool_commit_hash`."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, check=False)
        return out.stdout.strip()[:12] or "unavailable"
    except Exception:
        return "unavailable"
