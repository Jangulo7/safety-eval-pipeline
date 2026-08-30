"""The reproducibility gate: refuse to start a run that cannot produce comparable numbers.

``doctor`` answers *can this run at all* — credentials, datasets, provider extras. This
module answers a different and harder question: **if it runs, will the numbers mean
anything?**

A benchmark score is comparable across models only when every model saw the same items,
under the same generation parameters, graded by the same judge, at the same benchmark
version. Any one of those breaking turns a leaderboard into a ranking of experimental
artefacts. None of them announce themselves at runtime — a provider that discards `seed`
returns a perfectly ordinary response — so they are checked before the spend and again
against the recorded run.

The unit of comparison is **the benchmark, not the run**. StrongREJECT is published at
temperature 0.75 and XSTest at 0.0; forcing one number on both would either break
StrongREJECT's protocol or make XSTest nondeterministic. So the gate requires parameters to
be identical *within* a benchmark and merely *disclosed* between benchmarks.

Every failure carries the correction, not just the complaint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .catalog import Catalog
from .config import RunConfig
from .results import CellStatus, ResultSet


class Severity(str, Enum):
    BLOCK = "block"
    """Reproducibility is broken. The run must not start, or its results must not be ranked."""

    WARN = "warn"
    """Comparable, but a caveat must travel with the numbers."""

    OK = "ok"


@dataclass
class Issue:
    """One reproducibility defect, with the correction that fixes it."""

    id: str
    severity: Severity
    scope: str
    """The benchmark it applies to, or ``run`` for a run-wide condition."""

    problem: str
    correction: str
    evidence: str = ""

    def render(self, width: int = 92) -> str:
        import textwrap

        mark = {Severity.BLOCK: "BLOCKED", Severity.WARN: "WARN", Severity.OK: "ok"}[
            self.severity
        ]
        out = [f"  [{mark}] {self.scope} · {self.id}"]
        for label, text in (("", self.problem), ("evidence: ", self.evidence),
                            ("FIX: ", self.correction)):
            if not text:
                continue
            wrapped = textwrap.wrap(f"{label}{' '.join(text.split())}", width=width)
            out += [f"         {line}" for line in wrapped]
        return "\n".join(out)


@dataclass
class Verdict:
    """The gate's decision."""

    issues: list[Issue] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.issues.append(Issue(**kwargs))

    @property
    def blockers(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.BLOCK]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARN]

    @property
    def reproducible(self) -> bool:
        return not self.blockers

    @property
    def exit_code(self) -> int:
        return 0 if self.reproducible else 1

    def render(self) -> str:
        lines = ["", "reproducibility gate", ""]
        if not self.issues:
            lines.append("  All models will be run under identical conditions within every "
                         "benchmark.")
        else:
            lines += [i.render() for i in self.issues]
            lines.append("")
        lines.append(
            f"  {len(self.checked)} condition(s) checked · {len(self.blockers)} blocking · "
            f"{len(self.warnings)} warning(s)")
        if self.blockers:
            lines += [
                "",
                "  The run was NOT started. Correct the items marked BLOCKED above and",
                "  re-run. Each one breaks the claim that the models were measured under",
                "  the same conditions, which is what makes their scores comparable.",
            ]
        return "\n".join(lines)


# ------------------------------------------------------------------------------ pre-run

def check_config(
    config: RunConfig,
    catalog: Catalog | None = None,
    *,
    parameter_support: dict[str, set[str]] | None = None,
) -> Verdict:
    """Check a configuration before anything is spent.

    Args:
        config: the run configuration.
        catalog: the benchmark catalog.
        parameter_support: optional map of model id -> the generation parameters that model's
            provider actually accepts. When supplied, a parameter the pipeline sends but a
            provider silently discards is a blocker: it means the models were not run under
            the conditions the report would claim.
    """
    catalog = catalog or config.catalog
    v = Verdict()

    _check_model_set(v, config)
    _check_sample_selection(v, config, catalog)
    _check_per_benchmark_params(v, config, catalog)
    _check_grader(v, config)
    _check_parameter_support(v, config, parameter_support)
    return v


def _check_model_set(v: Verdict, config: RunConfig) -> None:
    v.checked.append("model set")
    if len(config.models) < 2:
        return

    providers = {m.provider for m in config.models}
    if len(providers) > 1:
        v.add(id="mixed-providers", severity=Severity.WARN, scope="run",
              problem="Models are served by more than one provider, so the serving stack "
                      "is not held constant across the comparison.",
              evidence=f"providers: {sorted(providers)}",
              correction="Serve every model through one provider — locally through vLLM, "
                         "or through one hosted operator — or report per-provider results "
                         "as separate tables rather than one ranking.")

    routed = [m.id for m in config.models if m.provider == "openrouter"]
    pinned = config.raw.get("provider_routing") or {}
    if routed and not pinned:
        v.add(id="unpinned-routing", severity=Severity.BLOCK, scope="run",
              problem="OpenRouter models are not pinned to a provider or a numeric "
                      "precision. The router chooses per request, so the same model id can "
                      "be served at fp8 in one cell and bf16 in another — the scores would "
                      "not be of one artifact.",
              evidence=f"unpinned: {routed}",
              correction="Add `provider_routing: {order: [<Provider>], "
                         "allow_fallbacks: false, quantizations: [<fmt>]}` to "
                         "config/eval_config.yaml, or move to locally served models where "
                         "the precision is yours to set.")


def _check_sample_selection(v: Verdict, config: RunConfig, catalog: Catalog) -> None:
    v.checked.append("sample selection")
    d = config.defaults

    capped = [t for t in config.tasks if config.limit_for(t) is not None]
    if capped and d.sample_shuffle is None:
        v.add(id="unseeded-selection", severity=Severity.BLOCK, scope="run",
              problem="A sample cap is set with no dataset-order seed, so the run takes the "
                      "head of each dataset. These datasets are grouped by category, so the "
                      "cap would evaluate one or two categories and the subset would not be "
                      "reproducible.",
              evidence=f"capped tasks: {[t.key for t in capped]}, sample_shuffle=None",
              correction="Set `defaults.sample_shuffle: 42`. Note this is not `seed`, which "
                         "seeds generation and has no effect on which samples are drawn.")

    for task in config.tasks:
        bench = catalog[task.benchmark]
        if bench.order_is_nondeterministic and not bench.order_pinned_by(task.args):
            required = bench.sample_order.get("deterministic_when") or {}
            v.add(id="unpinned-shuffle", severity=Severity.BLOCK, scope=task.key,
                  problem="The task shuffles its dataset with an unseeded shuffle, so each "
                          "model would be scored on a different set of prompts.",
                  evidence="two loads of 50 samples were measured to share none",
                  correction=f"add {required} to this task's `args` in "
                             "config/eval_config.yaml")

        strata = int((bench.subsets.get(task.subset or "", {}) or {}).get("strata")
                     or bench.dataset.get("strata") or 0)
        limit = config.limit_for(task)
        if strata and limit is not None and limit < strata:
            v.add(id="cap-below-strata", severity=Severity.BLOCK, scope=task.key,
                  problem=f"A cap of {limit} cannot cover {strata} dataset strata however "
                          "it is shuffled, so whole categories go unevaluated.",
                  correction=f"raise `defaults.limit` to at least {strata * 5}, or set it to "
                             "`null` to run the full dataset")


def _check_per_benchmark_params(v: Verdict, config: RunConfig, catalog: Catalog) -> None:
    """Generation parameters must be identical within a benchmark, and declared."""
    v.checked.append("per-benchmark generation parameters")
    for task in config.tasks:
        bench = catalog[task.benchmark]
        if not bench.protocol:
            v.add(id="undeclared-params", severity=Severity.BLOCK, scope=task.key,
                  problem="The benchmark declares no generation protocol, so the parameters "
                          "applied would come from whatever the serving provider defaults "
                          "to — which differs between providers and is not recorded.",
                  correction="add a `protocol:` block to this benchmark in "
                             "config/benchmarks.yaml giving temperature, max_tokens and "
                             "epochs, each marked `source: task` or `source: pipeline`")
            continue

        temperature = (bench.protocol.get("temperature") or {}).get("value")
        if temperature and temperature > 0 and config.defaults.seed is None:
            v.add(id="unseeded-sampling", severity=Severity.BLOCK, scope=task.key,
                  problem=f"The protocol samples at temperature {temperature} with no seed, "
                          "so the run cannot be reproduced.",
                  correction="set `defaults.seed`, and confirm every model's provider "
                             "honours it — a provider that discards `seed` makes a "
                             "temperature above 0 irreproducible whatever the config says")

        for key, spec in bench.protocol.items():
            if spec.get("source") == "pipeline" and not spec.get("note"):
                v.add(id="undisclosed-choice", severity=Severity.WARN, scope=task.key,
                      problem=f"`{key}` is a pipeline choice rather than the benchmark's "
                              "protocol, and carries no note explaining it.",
                      correction=f"add a `note:` to `{key}` in config/benchmarks.yaml "
                                 "so the report can say why this value was chosen")


def _check_grader(v: Verdict, config: RunConfig) -> None:
    """One judge for every model of a benchmark, or the scores are not comparable."""
    v.checked.append("grader")
    graders: dict[str, set[str]] = {}
    for task in config.tasks:
        bench = config.catalog[task.benchmark]
        graders.setdefault(task.key, set()).add(
            str(task.args.get(bench.grader_kwarg, config.grader_model))
        )
    for task_key, values in graders.items():
        if len(values) > 1:
            v.add(id="mixed-grader", severity=Severity.BLOCK, scope=task_key,
                  problem="More than one grader is configured for this benchmark. "
                          "Judge-graded scores are joint properties of the model and the "
                          "judge, so they would not be comparable.",
                  evidence=f"graders: {sorted(values)}",
                  correction="use one grader for every model of a benchmark")


def _check_parameter_support(
    v: Verdict, config: RunConfig, support: dict[str, set[str]] | None
) -> None:
    """A parameter the provider silently discards is worse than one never sent."""
    if not support:
        return
    v.checked.append("provider parameter support")

    sending = {"seed"} | {
        key for bench in config.catalog for key in bench.protocol if key != "epochs"
    }
    for model in config.models:
        accepted = support.get(model.id)
        if accepted is None:
            continue
        discarded = sorted(sending - accepted)
        if discarded:
            v.add(id="discarded-parameters", severity=Severity.BLOCK, scope="run",
                  problem=f"{model.label} is served by a provider that does not accept "
                          f"{discarded}. The pipeline would send them, the provider would "
                          "silently drop them, and the report would claim conditions that "
                          "were never applied to this model.",
                  evidence=f"{model.id} accepts {sorted(accepted)}",
                  correction="either drop these parameters for every model so the "
                             "comparison stays symmetric, or replace this model with one "
                             "whose provider honours them. Do not leave the run asymmetric.")


# ----------------------------------------------------------------------------- post-run

def check_results(results: ResultSet, config: RunConfig) -> Verdict:
    """Check a completed run against what actually happened.

    The pre-run check reads intent; this one reads evidence. A provider that ignored a
    parameter, a dataset that changed under the run, or a benchmark version that moved
    between cells all show up only here.
    """
    v = Verdict()
    v.checked.append("recorded conditions")

    for task_key in results.task_keys:
        cells = [c for c in results if c.task_key == task_key and c.status is CellStatus.OK]
        if len(cells) < 2:
            continue

        for label, values in (
            ("generation parameters",
             {repr(sorted(c.applied_generate_config.items())) for c in cells}),
            ("dataset fingerprint", {str(c.dataset_fingerprint) for c in cells}),
            ("benchmark version", {str(c.full_task_version) for c in cells}),
            ("grader", {str(c.grader_model) for c in cells}),
            ("sample cap", {str(c.n_requested) for c in cells}),
            ("dataset-order seed", {str(c.sample_shuffle) for c in cells}),
        ):
            if len(values) > 1:
                v.add(id=label.replace(" ", "-"), severity=Severity.BLOCK, scope=task_key,
                      problem=f"The {label} was not the same for every model, so these "
                              "scores are not a like-for-like comparison.",
                      evidence="; ".join(sorted(values))[:300],
                      correction="re-run the differing cells under the conditions of the "
                                 "others, or report them as separate tables rather than "
                                 "one ranking")

        for cell in cells:
            applied = cell.applied_generate_config
            if not applied:
                continue
            expected = config.catalog[cell.benchmark].generate_params()
            for key, want in expected.items():
                got = applied.get(key)
                if got is not None and got != want:
                    v.add(id="parameter-not-applied", severity=Severity.BLOCK,
                          scope=f"{task_key}/{cell.label}",
                          problem=f"`{key}` was configured as {want} but the harness "
                                  f"recorded {got} as applied.",
                          correction="reconcile config/benchmarks.yaml with the harness; "
                                     "the recorded value is what produced the score")

    return v
