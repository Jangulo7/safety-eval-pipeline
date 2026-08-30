"""Run configuration: models, tasks, gates, leaderboard weights.

Loaded from ``config/eval_config.yaml``. Nothing here is hardcoded in Python — the point is
that a reviewer can read one YAML file and know exactly what was run.

Validation is deliberately strict and happens at load time: a metric reference that does not
exist in the catalog, a gate with no bound, an unresolvable ``${}`` interpolation, or a task
naming an unknown benchmark are all startup errors. The alternative — discovering a typo
after spending an hour of provider credit — is worse.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .catalog import Catalog, CatalogError, MetricSpec

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "eval_config.yaml"

_INTERPOLATION = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


class ConfigError(ValueError):
    """The run configuration is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    """A model under evaluation."""

    id: str
    family: str
    label: str

    @property
    def provider(self) -> str:
        """The Inspect provider prefix, e.g. ``openrouter``."""
        return self.id.split("/", 1)[0]

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier used for log directories."""
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", self.id).strip("-")


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark configured for a run.

    ``key`` is the run-level name (``xstest_safe``) and may differ from ``benchmark``
    (``xstest``) because one benchmark can be run under several configurations.
    """

    key: str
    benchmark: str
    args: dict[str, Any] = field(default_factory=dict)
    subset: str | None = None

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", self.key).strip("-")


@dataclass(frozen=True)
class GateSpec:
    """A release threshold on one metric of one task.

    Bounds are in the metric's *native* units, which differ between benchmarks: XSTest is a
    percentage, StrongREJECT's headline metric is 0-5. ``validate`` cross-checks each bound
    against the catalog range so that a gate which can never fire is rejected at load time
    rather than silently passing forever.
    """

    id: str
    task: str
    metric: str
    min: float | None = None
    max: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ConfigError(f"gate {self.id!r} has neither a min nor a max bound")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ConfigError(f"gate {self.id!r} has min {self.min} > max {self.max}")


@dataclass(frozen=True)
class Defaults:
    """Decoding and concurrency parameters applied to every cell."""

    limit: int = 50
    max_connections: int = 8
    temperature: float = 0.0
    seed: int = 42
    max_retries: int = 1
    timeout_s: int = 1800


@dataclass(frozen=True)
class LeaderboardSpec:
    """How the composite index is built. Printed above every table that uses it."""

    index_name: str
    weights: dict[str, float]
    tie_on_overlapping_ci: bool = True

    @property
    def normalised_weights(self) -> dict[str, float]:
        """Weights rescaled to sum to 1, so a partial matrix still produces a comparable index."""
        total = sum(self.weights.values())
        if total <= 0:
            raise ConfigError("leaderboard weights must sum to a positive number")
        return {k: v / total for k, v in self.weights.items()}


@dataclass(frozen=True)
class OutputSpec:
    """Which artefacts to produce and where."""

    results_dir: Path = Path("results")
    log_dir: Path = Path("logs")
    charts: bool = True
    html: bool = True
    pdf: bool = True


class RunConfig:
    """A validated run configuration."""

    def __init__(self, data: dict[str, Any], catalog: Catalog, source: Path | None = None) -> None:
        self.source = source
        self.catalog = catalog
        self.provider: str = data.get("provider", "openrouter")
        self.grader_model: str = data["grader_model"]
        self.raw = data
        scoped = data.get("_scoped") or {}
        self.dropped_gates: list[str] = list(scoped.get("dropped_gates", []))
        """Gates removed because ``--tasks`` scoped the run away from them. Reported rather
        than silently omitted, so a green gate report cannot imply coverage it does not have."""
        self.dropped_weights: list[str] = list(scoped.get("dropped_weights", []))

        self.models = [
            ModelSpec(id=m["id"], family=m.get("family", m["id"].split("/")[1]),
                      label=m.get("label", m["id"].rsplit("/", 1)[-1]))
            for m in data["models"]
        ]
        if not self.models:
            raise ConfigError("no models configured")

        self.tasks = [
            TaskSpec(key=t["key"], benchmark=t["benchmark"], args=dict(t.get("args") or {}),
                     subset=t.get("subset"))
            for t in data["tasks"]
        ]
        if not self.tasks:
            raise ConfigError("no tasks configured")

        d = data.get("defaults") or {}
        self.defaults = Defaults(**{k: v for k, v in d.items() if k in Defaults.__annotations__})

        self.gates = [GateSpec(**g) for g in (data.get("gates") or [])]

        lb = data.get("leaderboard") or {}
        self.leaderboard = LeaderboardSpec(
            index_name=lb.get("index_name", "Safety Index"),
            weights={k: float(v) for k, v in (lb.get("weights") or {}).items()},
            tie_on_overlapping_ci=bool(lb.get("tie_on_overlapping_ci", True)),
        )

        o = data.get("output") or {}
        self.output = OutputSpec(
            results_dir=Path(o.get("results_dir", "results")),
            log_dir=Path(o.get("log_dir", "logs")),
            charts=bool(o.get("charts", True)),
            html=bool(o.get("html", True)),
            pdf=bool(o.get("pdf", True)),
        )

        self.validate()

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        catalog: Catalog | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> RunConfig:
        """Load, interpolate and validate a run configuration."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"run configuration not found at {path}")
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"{path} does not contain a mapping")
        data = interpolate(data)
        if overrides:
            data = _apply_overrides(data, overrides)
        return cls(data, catalog or Catalog.load(), source=path)

    # --------------------------------------------------------------- validation

    def task(self, key: str) -> TaskSpec:
        """Look up a configured task by key."""
        for t in self.tasks:
            if t.key == key:
                return t
        raise ConfigError(f"unknown task {key!r}; configured: {[t.key for t in self.tasks]}")

    def metric_for(self, task_key: str, metric_address: str) -> MetricSpec:
        """Resolve a metric address against the benchmark behind a task key."""
        bench = self.catalog[self.task(task_key).benchmark]
        return bench.metric(metric_address)

    def resolve_reference(self, reference: str) -> tuple[TaskSpec, MetricSpec]:
        """Resolve a ``"<task_key>:<score>/<metric>"`` reference used in gates and weights."""
        if ":" not in reference:
            raise ConfigError(
                f"metric reference {reference!r} must be '<task_key>:<score_name>/<metric_key>'"
            )
        task_key, addr = reference.split(":", 1)
        task = self.task(task_key)
        return task, self.metric_for(task_key, addr)

    def validate(self) -> None:
        """Fail loudly on anything that would waste a run."""
        seen: set[str] = set()
        for t in self.tasks:
            if t.key in seen:
                raise ConfigError(f"duplicate task key {t.key!r}")
            seen.add(t.key)
            if t.benchmark not in self.catalog:
                raise ConfigError(
                    f"task {t.key!r} names unknown benchmark {t.benchmark!r}; "
                    f"known: {sorted(self.catalog.benchmarks)}"
                )
            bench = self.catalog[t.benchmark]
            if bench.subsets and t.subset and t.subset not in bench.subsets:
                raise ConfigError(
                    f"task {t.key!r}: benchmark {t.benchmark!r} has no subset {t.subset!r}; "
                    f"known: {sorted(bench.subsets)}"
                )
            # The grader kwarg differs per benchmark (scorer_model vs judge_llm). Wiring the
            # wrong one is silent: Inspect falls back to its default grader and the numbers
            # are quietly ungoverned.
            grader_kwargs = {"scorer_model", "judge_llm", "grader_model"}
            supplied = grader_kwargs & set(t.args)
            wrong = supplied - {bench.grader_kwarg}
            if wrong:
                raise ConfigError(
                    f"task {t.key!r} passes {sorted(wrong)} but benchmark {t.benchmark!r} "
                    f"takes its grader as {bench.grader_kwarg!r}. Passing the wrong kwarg is "
                    "silently ignored by Inspect and the task would grade with its default "
                    "model instead."
                )

        for g in self.gates:
            try:
                metric = self.resolve_reference(f"{g.task}:{g.metric}")[1]
            except (ConfigError, CatalogError) as exc:
                raise ConfigError(f"gate {g.id!r}: {exc}") from None
            lo, hi = metric.range
            for bound_name, bound in (("min", g.min), ("max", g.max)):
                if bound is None:
                    continue
                if not lo <= bound <= hi:
                    raise ConfigError(
                        f"gate {g.id!r}: {bound_name}={bound} is outside the range of "
                        f"{metric.address} ({lo}-{hi}"
                        f"{', a percentage' if metric.unit == 'percent' else ''}). "
                        "A bound outside the metric's range can never fire."
                    )

        if suspicious := self.suspicious_bounds():
            raise ConfigError(
                "gate bounds look like a unit mistake: "
                + "; ".join(suspicious)
                + ". Percentage metrics take bounds on a 0-100 scale. If a sub-1% bound is "
                "genuinely intended, express it as a range with an explicit min as well."
            )

        for ref in self.leaderboard.weights:
            try:
                self.resolve_reference(ref)
            except (ConfigError, CatalogError) as exc:
                raise ConfigError(f"leaderboard weight {ref!r}: {exc}") from None

    def suspicious_bounds(self) -> list[str]:
        """Bounds that are in range but almost certainly the wrong unit.

        The failure this catches is specific and expensive: XSTest's ``refusal_rate`` is a
        percentage on 0-100, so a gate written as ``max: 0.20`` — the natural thing to write
        if you think in fractions — is technically valid and can essentially never fire. A
        gate that cannot fire is worse than no gate, because it looks like coverage.
        """
        out: list[str] = []
        for g in self.gates:
            try:
                metric = self.resolve_reference(f"{g.task}:{g.metric}")[1]
            except (ConfigError, CatalogError):
                continue
            if metric.unit != "percent" or metric.range[1] < 100:
                continue
            for name, bound in (("min", g.min), ("max", g.max)):
                if bound is not None and 0.0 < bound < 1.0 and not (g.min and g.max):
                    out.append(
                        f"{g.id}.{name}={bound} on {metric.address}, which is a percentage "
                        f"(0-{metric.range[1]:g}) — did you mean {bound * 100:g}?"
                    )
        return out

    # ------------------------------------------------------------------ helpers

    @property
    def cells(self) -> list[tuple[ModelSpec, TaskSpec]]:
        """Every (model, task) pair the run will attempt, in execution order."""
        return [(m, t) for t in self.tasks for m in self.models]

    def estimated_samples(self) -> int:
        """Total graded samples the run will request. Shown before spending anything."""
        return len(self.cells) * self.defaults.limit


def interpolate(data: Any, root: dict[str, Any] | None = None) -> Any:
    """Resolve ``${dotted.path}`` references against the config root.

    Also resolves ``${env:VAR}``. An unresolvable reference is an error rather than being
    left as a literal, because a literal ``"${grader_model}"`` passed to Inspect as a model
    name fails much later and much less clearly.
    """
    if root is None:
        root = data if isinstance(data, dict) else {}

    if isinstance(data, dict):
        return {k: interpolate(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [interpolate(v, root) for v in data]
    if not isinstance(data, str):
        return data

    def replace(match: re.Match[str]) -> str:
        path = match.group(1)
        if path.startswith("env."):
            var = path[4:]
            value = os.environ.get(var)
            if value is None:
                raise ConfigError(f"${{{path}}} refers to unset environment variable {var}")
            return value
        node: Any = root
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(
                    f"${{{path}}} does not resolve against the configuration root"
                )
            node = node[part]
        if not isinstance(node, (str, int, float)):
            raise ConfigError(f"${{{path}}} resolves to a {type(node).__name__}, not a scalar")
        return str(node)

    return _INTERPOLATION.sub(replace, data)


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI/UI overrides onto loaded config data.

    Supported keys: ``models`` (list of ids), ``tasks`` (list of keys), and any scalar in
    ``defaults``. Filtering to an unknown id or key is an error, not a silent empty matrix.
    """
    data = dict(data)
    if models := overrides.get("models"):
        wanted = set(models)
        kept = [m for m in data["models"] if m["id"] in wanted or m.get("label") in wanted]
        missing = wanted - {m["id"] for m in kept} - {m.get("label") for m in kept}
        if missing:
            raise ConfigError(
                f"--models named {sorted(missing)}, which are not in "
                f"{[m['id'] for m in data['models']]}"
            )
        data["models"] = kept
    if tasks := overrides.get("tasks"):
        wanted = set(tasks)
        kept = [t for t in data["tasks"] if t["key"] in wanted]
        missing = wanted - {t["key"] for t in kept}
        if missing:
            raise ConfigError(
                f"--tasks named {sorted(missing)}, which are not in "
                f"{[t['key'] for t in data['tasks']]}"
            )
        data["tasks"] = kept
        # A subset run cannot evaluate a gate on a task it did not run. Those gates are
        # dropped rather than left to fail as "cell was not run", and the drop is recorded
        # so the gate report can say the run was scoped rather than implying full coverage.
        dropped_gates = [g["id"] for g in (data.get("gates") or []) if g["task"] not in wanted]
        data["gates"] = [g for g in (data.get("gates") or []) if g["task"] in wanted]
        weights = (data.get("leaderboard") or {}).get("weights") or {}
        dropped_weights = [k for k in weights if k.split(":", 1)[0] not in wanted]
        if dropped_weights:
            data.setdefault("leaderboard", {})["weights"] = {
                k: v for k, v in weights.items() if k.split(":", 1)[0] in wanted
            }
        data["_scoped"] = {"dropped_gates": dropped_gates,
                           "dropped_weights": dropped_weights}
    for key, value in (overrides.get("defaults") or {}).items():
        if value is not None:
            data.setdefault("defaults", {})[key] = value
    for key in ("charts", "html", "pdf"):
        if (value := overrides.get(key)) is not None:
            data.setdefault("output", {})[key] = value
    return data
