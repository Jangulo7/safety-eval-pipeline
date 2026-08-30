"""The benchmark catalog: what each benchmark and metric *means*.

Loaded from ``config/benchmarks.yaml``. This module is deliberately the only place that
knows a metric's range, direction and plain-English explanation, so that a gate, a chart
axis, a leaderboard column and a PDF page can never disagree about them.

Metric addressing
-----------------
Inspect namespaces metrics per *score*, and some metric keys are themselves qualified. A
sycophancy log, for example, carries::

    scores[0].name = "sycophancy_scorer"  metrics = {"confidence", "apologize_rate"}
    scores[3].name = "truthfulness"       metrics = {"inspect_evals/truthfulness"}

so the only unambiguous address is ``"<score_name>/<metric_key>"``. Bare keys are accepted
as a convenience but resolve to an error when more than one score exposes them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "benchmarks.yaml"


class Direction(str, Enum):
    """Which way is good for a metric."""

    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    CONTEXT_DEPENDENT = "context_dependent"


class MetricKind(str, Enum):
    """How a metric's uncertainty should be estimated.

    ``RATE`` is a proportion of a countable denominator and gets a Wilson interval.
    ``BOUNDED`` is a mean of per-sample values on a known interval and gets a bootstrap.
    """

    RATE = "rate"
    BOUNDED = "bounded"


class CatalogError(ValueError):
    """The catalog is malformed, or a lookup into it failed."""


@dataclass(frozen=True)
class MetricSpec:
    """One metric of one benchmark, with everything needed to read it correctly."""

    address: str
    """``"<score_name>/<metric_key>"`` as it appears in an Inspect log."""

    label: str
    short: str
    range: tuple[float, float]
    direction: Direction
    kind: MetricKind
    explain: str
    denominator: str = "scored_samples"
    unit: str | None = None
    primary: bool = False
    capability_control: bool = False
    direction_by_subset: dict[str, Direction] = field(default_factory=dict)

    @property
    def score_name(self) -> str:
        """The Inspect score this metric lives under."""
        return self.address.split("/", 1)[0]

    @property
    def metric_key(self) -> str:
        """The metric key within that score (may itself contain a slash)."""
        return self.address.split("/", 1)[1]

    def direction_for(self, subset: str | None = None) -> Direction:
        """Resolve the direction, which for XSTest depends on which subset was run."""
        if self.direction is not Direction.CONTEXT_DEPENDENT:
            return self.direction
        if subset is None or subset not in self.direction_by_subset:
            raise CatalogError(
                f"metric {self.address!r} is context-dependent and needs a subset to "
                f"resolve its direction; got {subset!r}. Known subsets: "
                f"{sorted(self.direction_by_subset)}"
            )
        return self.direction_by_subset[subset]

    def normalise(self, value: float, subset: str | None = None) -> float:
        """Map ``value`` onto 0-1 where 1 is always good.

        This is what makes a cross-benchmark composite index possible at all: XSTest is a
        percentage where low is good on the safe subset and high is good on the unsafe one,
        StrongREJECT is 0-5 where low is good. Without an explicit range and direction per
        metric a composite is meaningless.

        ``nan`` propagates: a metric the grader failed to produce must not become a 0.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return math.nan
        lo, hi = self.range
        if hi == lo:
            raise CatalogError(f"metric {self.address!r} has a degenerate range {self.range}")
        scaled = (value - lo) / (hi - lo)
        scaled = min(1.0, max(0.0, scaled))
        return scaled if self.direction_for(subset) is Direction.HIGHER_BETTER else 1.0 - scaled


@dataclass(frozen=True)
class Caveat:
    """A known trap in reading a benchmark, surfaced in the UI and the report."""

    id: str
    severity: str
    text: str


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark: its Inspect task, its metrics, and how to read them."""

    key: str
    task: str
    title: str
    grader_kwarg: str
    """``scorer_model`` for xstest/sycophancy, ``judge_llm`` for strong_reject. Wiring one
    UI control to the wrong kwarg silently grades with the default model instead."""

    publish_logs: bool
    """False means transcripts are never written to a published artefact. See SPEC §9."""

    interpretation: dict[str, str]
    metrics: dict[str, MetricSpec]
    dataset: dict[str, Any] = field(default_factory=dict)
    subsets: dict[str, dict[str, Any]] = field(default_factory=dict)
    caveats: tuple[Caveat, ...] = ()
    task_version_expected: str | None = None
    dataset_samples: int | None = None

    @property
    def gated(self) -> bool:
        """Whether the dataset needs credentialed access before any run can start."""
        return bool(self.dataset.get("gated", False))

    @property
    def primary_metrics(self) -> list[MetricSpec]:
        """The metrics a summary view should lead with."""
        return [m for m in self.metrics.values() if m.primary]

    def metric(self, address: str) -> MetricSpec:
        """Look up a metric by full address, or by bare key when that is unambiguous."""
        if address in self.metrics:
            return self.metrics[address]
        matches = [m for m in self.metrics.values() if m.metric_key == address or m.short == address]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise CatalogError(
                f"benchmark {self.key!r} has no metric {address!r}; "
                f"known: {sorted(self.metrics)}"
            )
        raise CatalogError(
            f"metric {address!r} is ambiguous in benchmark {self.key!r} — it appears under "
            f"scores {sorted(m.score_name for m in matches)}. Use the full "
            f"'<score_name>/<metric_key>' address."
        )


@dataclass(frozen=True)
class TradeoffSpec:
    """A pair of metrics that must be read against each other, not ranked."""

    id: str
    title: str
    x: dict[str, Any]
    y: dict[str, Any]
    caption: str


class Catalog:
    """The parsed benchmark catalog."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._raw = data
        self.verified_against: dict[str, str] = data.get("verified_against", {})
        self.benchmarks: dict[str, BenchmarkSpec] = {
            key: _parse_benchmark(key, spec) for key, spec in data["benchmarks"].items()
        }
        self.tradeoffs: dict[str, TradeoffSpec] = {
            t["id"]: TradeoffSpec(id=t["id"], title=t["title"], x=t["x"], y=t["y"],
                                  caption=t["caption"].strip())
            for t in data.get("tradeoffs", [])
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> Catalog:
        """Load the catalog from YAML."""
        path = Path(path) if path else DEFAULT_CATALOG_PATH
        if not path.exists():
            raise CatalogError(f"benchmark catalog not found at {path}")
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or "benchmarks" not in data:
            raise CatalogError(f"{path} is not a benchmark catalog (no 'benchmarks' key)")
        return cls(data)

    def __getitem__(self, key: str) -> BenchmarkSpec:
        try:
            return self.benchmarks[key]
        except KeyError:
            raise CatalogError(
                f"unknown benchmark {key!r}; known: {sorted(self.benchmarks)}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self.benchmarks

    def __iter__(self) -> Iterator[BenchmarkSpec]:
        return iter(self.benchmarks.values())

    def __len__(self) -> int:
        return len(self.benchmarks)

    def resolve(self, reference: str) -> tuple[BenchmarkSpec, MetricSpec]:
        """Resolve a ``"<benchmark>:<score>/<metric>"`` reference used in configs.

        The benchmark part may be a *task key* (``xstest_safe``) rather than a catalog key
        (``xstest``); the caller maps task keys to benchmarks and passes the catalog key.
        """
        if ":" not in reference:
            raise CatalogError(
                f"metric reference {reference!r} must be '<benchmark>:<score>/<metric>'"
            )
        bench_key, metric_addr = reference.split(":", 1)
        bench = self[bench_key]
        return bench, bench.metric(metric_addr)


def _parse_direction(value: str, where: str) -> Direction:
    try:
        return Direction(value)
    except ValueError:
        raise CatalogError(
            f"{where}: unknown direction {value!r}; expected one of "
            f"{[d.value for d in Direction]}"
        ) from None


def _parse_benchmark(key: str, spec: dict[str, Any]) -> BenchmarkSpec:
    for required in ("task", "title", "grader_kwarg", "metrics", "interpretation"):
        if required not in spec:
            raise CatalogError(f"benchmark {key!r} is missing required field {required!r}")

    metrics: dict[str, MetricSpec] = {}
    for address, m in spec["metrics"].items():
        where = f"benchmark {key!r} metric {address!r}"
        if "/" not in address:
            raise CatalogError(
                f"{where}: address must be '<score_name>/<metric_key>' — Inspect namespaces "
                "metrics per score and a bare key is ambiguous"
            )
        for required in ("label", "range", "direction", "kind", "explain"):
            if required not in m:
                raise CatalogError(f"{where}: missing required field {required!r}")
        rng = m["range"]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise CatalogError(f"{where}: range must be a two-element [lo, hi]")
        direction = _parse_direction(m["direction"], where)
        by_subset = {
            s: _parse_direction(d, where)
            for s, d in (m.get("direction_by_subset") or {}).items()
        }
        if direction is Direction.CONTEXT_DEPENDENT and not by_subset:
            raise CatalogError(
                f"{where}: direction is context_dependent but direction_by_subset is empty, "
                "so the direction can never be resolved"
            )
        try:
            kind = MetricKind(m["kind"])
        except ValueError:
            raise CatalogError(
                f"{where}: unknown kind {m['kind']!r}; expected one of "
                f"{[k.value for k in MetricKind]}"
            ) from None
        metrics[address] = MetricSpec(
            address=address,
            label=m["label"],
            short=m.get("short", address.rsplit("/", 1)[-1]),
            range=(float(rng[0]), float(rng[1])),
            direction=direction,
            direction_by_subset=by_subset,
            kind=kind,
            explain=" ".join(m["explain"].split()),
            denominator=m.get("denominator", "scored_samples"),
            unit=m.get("unit"),
            primary=bool(m.get("primary", False)),
            capability_control=bool(m.get("capability_control", False)),
        )

    return BenchmarkSpec(
        key=key,
        task=spec["task"],
        title=spec["title"],
        grader_kwarg=spec["grader_kwarg"],
        publish_logs=bool(spec.get("publish_logs", False)),
        interpretation={k: " ".join(v.split()) for k, v in spec["interpretation"].items()},
        metrics=metrics,
        dataset=spec.get("dataset", {}),
        subsets=spec.get("subsets", {}),
        caveats=tuple(
            Caveat(id=c["id"], severity=c.get("severity", "medium"),
                   text=" ".join(c["text"].split()))
            for c in spec.get("caveats", [])
        ),
        task_version_expected=spec.get("task_version_expected"),
        dataset_samples=spec.get("dataset_samples"),
    )
