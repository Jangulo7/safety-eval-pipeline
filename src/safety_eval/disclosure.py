"""Contamination disclosure and the Parameter Register, computed from the run itself.

Two disclosure frameworks travel with every report:

**Contamination disclosure** — four fields that
determine whether one evaluation's numbers can honestly be set beside another's: whether
results are broken down by stratum (F1), whether the elicitation conditions are
reproducible (F2), what contamination controls were applied (F3, five types), and whether
the instrument can be regenerated (F4). The codes are computed from what the run actually
recorded, not asserted — so a change that stops recording something lowers the score.

The schema is published at https://zenodo.org/records/21750019 and every artefact this
pipeline emits cites it beside the codes, because a code is meaningless without the rubric
that defines it.

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

DISCLOSURE_SCHEMA_URL = "https://zenodo.org/records/21750019"
DISCLOSURE_SCHEMA_CITATION = (
    "Coded against the contamination-disclosure schema, "
    f"{DISCLOSURE_SCHEMA_URL}: four fields (F1 strata reported, F2 elicitation budget, "
    "F3 contamination controls over five types, F4 regeneration). This pipeline computes "
    "the codes from what the run recorded rather than asserting them."
)

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
              "instruments are public and predate every model's training cutoff. Assume "
              "direct contamination. Named, uncontrolled."),
        Field("t2", "F3 · t2 Derivative",
              "2" if fingerprinted else "1",
              "Provenance tracked: each cell records the dataset source and Inspect's "
              "content fingerprint, so two runs can be shown to use the same items."
              if fingerprinted else
              "Sources are named but no content fingerprint is recorded."),
        Field("t3", "F3 · t3 Temporal", "2",
              f"Publication dates stated and related to the items: {dated}. All three "
              "predate every model's stated training cutoff, so contamination is likely "
              "rather than excluded."),
        Field("t4", "F3 · t4 Distributional", "0",
              "No perturbation, paraphrase-robustness or template-novelty testing was "
              "performed. Recorded as 0 rather than inflated."),
        Field("t5", "F3 · t5 Acquired", "2",
              "The solver chain is system_message then generate(). The evaluated model had "
              "no tools, no retrieval and no network access during scoring. Control stated; "
              "not established by transcript review."),
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
        "**t1 = 1 is the load-bearing code.** These benchmarks are public and predate every "
        "model's training cutoff, and no decontamination check ran. Assume some direct "
        "contamination. This does not void the measurement: a model that has seen XSTest "
        "and still over-refuses still over-refuses. It does rule out any claim of an "
        "uncontaminated result."
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

    provenance: str = "requested"
    """How the value was obtained, which decides how much it is worth.

    ``measured``     read back from the run's own artefacts: the eval log, the package
                     metadata, the git revision the harness recorded, the host.
    ``applied``      set by this pipeline, where no external system could ignore it --
                     dataset ordering, the sample cap, the grader we called.
    ``requested``    sent to an external system and *not* verified to have taken effect. A
                     provider that discarded it leaves this row unchanged.
    ``editorial``    a label the reporter chose. Not a measurement, and not posing as one.
    ``unavailable``  cannot be obtained with this code. Stated, never guessed.
    ``n/a``          does not apply to this serving arrangement, with the reason given.

    ``requested`` and ``measured`` are the load-bearing pair. Everything else exists so that
    no row has to be padded with a plausible-looking value.
    """


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

    def add(section: str, parameter: str, value: Any, status: str = "recorded",
            provenance: str = "requested") -> None:
        rows.append(RegisterRow(section, parameter, str(value), status, provenance))

    # A · model identity and quantization
    add("A · Model identity", "model_id", ", ".join(results.models), provenance="measured")
    add("A · Model identity", "base_model_family",
        ", ".join(sorted({c.family for c in results})), provenance="measured")
    weights = (first.serving if first else {}) or {}
    add("A · Model identity", "param_count_billions",
        f"{weights['param_count_billions']}B (computed from the checkpoint on disk)"
        if weights.get("param_count_billions") else "checkpoint not measured for this run",
        "recorded" if weights.get("param_count_billions") else "missing",
        provenance="measured" if weights.get("param_count_billions") else "unavailable")
    if local:
        # Declared, not verified. The pipeline asks vLLM to serve at this precision; it does
        # not read the precision back from the running server, so a server that ignored the
        # flag would leave this row unchanged.
        # Prefer what the server reported about itself over what we asked it for.
        reported = (first.serving if first else {}) or {}
        if reported.get("dtype") or reported.get("quantization"):
            add("A · Model identity", "quant_scheme",
                f"{reported.get('quantization', '?')} (dtype {reported.get('dtype', '?')}), "
                "reported by the server", provenance="measured")
            add("A · Model identity", "bits_per_weight",
                "16 (bfloat16)" if "bfloat16" in reported.get("dtype", "") else "see dtype",
                provenance="measured")
        else:
            add("A · Model identity", "quant_scheme",
                f"requested {serving.get('quantization', 'none')} "
                f"(dtype {serving.get('dtype', '?')}); the server did not report it back",
                "recorded", provenance="requested")
            add("A · Model identity", "bits_per_weight",
                f"{weights['bits_per_weight']} (from the checkpoint dtype "
                f"{weights.get('checkpoint_dtype', '?')})" if weights.get("bits_per_weight")
                else "checkpoint not measured for this run",
                "recorded" if weights.get("bits_per_weight") else "missing",
                provenance="measured" if weights.get("bits_per_weight") else "unavailable")
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
        add("A · Model identity", parameter, why, "not applicable", provenance="n/a")

    # B · inference configuration — per benchmark, because that is where it is held constant
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        cell = next((c for c in ok if c.task_key == task.key), None)
        applied = dict((cell.applied_generate_config if cell else {}) or {})
        # epochs is an eval-level setting, not a generate-config one; the harness records it
        # separately and reading it back is what makes it measured rather than restated.
        applied.update((cell.eval_config if cell else {}) or {})
        for key, spec in bench.protocol.items():
            source = "benchmark protocol" if spec.get("source") == "task" else "pipeline choice"
            # Prefer what the harness recorded as applied over what the catalog declared.
            # They should agree; where they do not, the applied value is what made the score.
            if key in applied:
                value = f"{applied[key]} ({source}, applied)"
                prov = "measured"
                if applied[key] != spec["value"]:
                    value += f" — DECLARED {spec['value']}, NOT APPLIED"
            else:
                # epochs is applied by the harness itself, so nothing external can drop it.
                # temperature and max_tokens are sent to a provider and may be ignored.
                # Applied by the harness itself, so nothing external could drop it. Named
                # distinctly: `local` above means "the model is served locally" and reusing
                # it here silently flipped the environment section to the hosted branch.
                harness_applied = key in {"epochs"}
                value = (f"{spec['value']} ({source}, "
                         f"{'applied' if harness_applied else 'requested, not read back'})")
                prov = "applied" if harness_applied else "requested"
            add("B · Inference", f"{task.key} · {key}", value, provenance=prov)
    seed_applied = (first.applied_generate_config.get("seed") if first else None)
    add("B · Inference", "random_seed · generation",
        f"{seed_applied} (applied)" if seed_applied is not None
        else f"{first.seed if first else '—'} (sent; provider did not report it back)",
        provenance="measured" if seed_applied is not None else "requested")
    shuffle_applied = (first.eval_config.get("sample_shuffle") if first else None)
    add("B · Inference", "random_seed · dataset order",
        f"{shuffle_applied} (applied, read back from the harness)" if shuffle_applied
        is not None else f"{first.sample_shuffle if first else '—'} (set, not read back)",
        provenance="measured" if shuffle_applied is not None else "applied")
    top_p_applied = (first.applied_generate_config.get("top_p") if first else None)
    add("B · Inference", "top_p",
        f"{top_p_applied} (applied)" if top_p_applied is not None
        else "not set for this run; the serving default applied and was not reported back",
        "recorded" if top_p_applied is not None else "missing",
        provenance="measured" if top_p_applied is not None else "unavailable")
    add("B · Inference", "top_k / repetition_penalty",
        "not reachable through this stack: Inspect's OpenAI-compatible provider does not "
        "forward either parameter, so the server default applies and cannot be set or read",
        "not applicable", provenance="n/a")
    served_ctx = (first.serving or {}).get("max_model_len") if first else None
    add("B · Inference", "max_context_window",
        f"{served_ctx} (reported by the server)" if served_ctx
        else (f"requested {serving.get('max_model_len')}; not reported back" if local
              else "provider default, not reported"),
        "recorded" if served_ctx else "missing",
        provenance="measured" if served_ctx else ("requested" if local else "unavailable"))
    applied_conc = (first.applied_generate_config.get("max_connections") if first else None)
    add("B · Inference", "batch_size (concurrency)",
        f"{applied_conc or (first.max_connections if first else '—')} concurrent requests",
        provenance="measured" if applied_conc else "applied")
    prompts = {c.task_key: (c.system_prompt or "none") for c in ok}
    add("B · Inference", "system_prompt",
        "; ".join(f"{k}: {v[:60]!r}" if v != "none" else f"{k}: none"
                  for k, v in sorted(prompts.items())) or "—",
        provenance="measured" if prompts else "unavailable")
    for parameter, why in (("rope_scaling", "model default; not overridden"),):
        add("B · Inference", parameter, why, "not applicable", provenance="n/a")

    # C · hardware and environment
    if local:
        engine = (first.serving or {}).get("engine_version") if first else None
        add("C · Environment", "inference_backend",
            f"vLLM {engine}" if engine else "vLLM (version not reported for this run)",
            "recorded" if engine else "missing",
            provenance="measured" if engine else "requested")
        # Read from the run's metadata where the runner captured it. Previously these rows
        # held the literal string "recorded at run time" while claiming status "recorded" --
        # a placeholder presented as a measurement, which is the exact failure this register
        # exists to prevent.
        host = (results.metadata.host or {}) if hasattr(results.metadata, "host") else {}
        for key, label in (("gpu_model", "gpu_model"), ("gpu_count", "gpu_count"),
                           ("driver_version", "driver_version"),
                           ("cuda_version", "cuda_version")):
            value = host.get(key)
            # `captured` is set when the host was read after the run rather than during it.
            # Accurate and not measured-at-run-time are different claims, and the row says
            # which one it is.
            note = host.get("captured")
            add("C · Environment", label,
                f"{value} ({note})" if value and note else (value or
                "not recorded for this run"),
                "recorded" if value else "missing",
                provenance="measured" if value else "unavailable")
    else:
        add("C · Environment", "inference_backend", "hosted API (provider-routed)")
        for parameter in ("gpu_model", "gpu_count", "cpu_model", "system_ram_gb",
                          "driver_version", "quant_backend_version"):
            add("C · Environment", parameter,
                "models run on the provider's hardware, which is not disclosed",
                "not applicable", provenance="n/a")

    # D · classification
    add("D · Classification", "benchmark_category",
        "Safety — over-refusal, under-refusal, sycophancy", provenance="editorial")
    add("D · Classification", "metric_type",
        "Refusal calibration; harmful-uplift; sycophancy. Not accuracy.",
        provenance="editorial")
    add("D · Classification", "dataset_version",
        "; ".join(sorted({f"{c.task_key} {c.full_task_version}" for c in ok
                          if c.full_task_version})) or "—", provenance="measured")


    # E · tool and pipeline metadata
    add("E · Pipeline", "eval_tool_version",
        f"inspect_ai {meta.inspect_ai_version}, inspect_evals {meta.inspect_evals_version}, "
        f"pipeline {meta.pipeline_version}", provenance="measured")
    # The commit the RUN happened at, from the harness's own record. `git rev-parse HEAD` at
    # report time answers a different question and was previously reporting a later commit.
    commit = next((c.run_commit for c in ok if c.run_commit), None)
    dirty = next((c.run_commit_dirty for c in ok if c.run_commit is not None), None)
    add("E · Pipeline", "eval_tool_commit_hash",
        f"{commit}{' (working tree DIRTY: the commit alone does not reproduce this run)' if dirty else ''}"
        if commit else "not recorded in the log",
        "recorded" if commit else "missing",
        provenance="measured" if commit else "unavailable")
    add("E · Pipeline", "run_timestamp", meta.started_utc, provenance="measured")
    add("E · Pipeline", "run_duration_seconds", f"{results.total_wall_clock_s:.0f}",
        provenance="measured")
    add("E · Pipeline", "dataset_fingerprint",
        "; ".join(sorted({f"{c.task_key}: {c.dataset_fingerprint}" for c in ok
                          if c.dataset_fingerprint})) or "—", provenance="measured")
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
