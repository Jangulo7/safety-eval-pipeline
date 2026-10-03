"""The record itself: fields, provenance marks, and the comparability tier they imply."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    """How a value was obtained, which decides what it is worth.

    Ordered from strongest to weakest. The ordering is used to compute a tier, so it is part
    of the contract rather than a presentational detail.
    """

    MEASURED = "measured"
    """Read back from the run's own artefacts: the log, the server, the package metadata.
    Nothing external could have discarded it without the record showing a different value."""

    APPLIED = "applied"
    """Set by the harness itself, where no external system was in a position to ignore it --
    dataset ordering, a sample cap, which judge was called."""

    REQUESTED = "requested"
    """Sent to an external system and **not** verified to have taken effect. A provider that
    silently dropped it leaves this field looking exactly as it would if it had been honoured.
    The weakest mark that still carries a value."""

    EDITORIAL = "editorial"
    """A label the reporter chose. Not a measurement and not posing as one."""

    UNAVAILABLE = "unavailable"
    """Cannot be obtained from this log by this code. Stated, never guessed."""

    NOT_APPLICABLE = "n/a"
    """Does not apply to this serving arrangement, with the reason carried on the field."""


# Only these two mean "the run established this". Everything else is a claim about intent,
# an absence, or a label.
ESTABLISHED = frozenset({Provenance.MEASURED, Provenance.APPLIED})


class Tier(str, Enum):
    """How far a record can support a comparison between runs.

    **Unvalidated.** This is a specification with a worked example, not a measured instrument:
    no study has shown that tier predicts reproducibility. It is published so that it can be
    argued with, and it is computed mechanically so that nobody has to take the author's word
    for which tier a record reached.
    """

    T0 = "0"
    """Scores only. The conditions are not recoverable, so the number cannot be compared with
    any other number."""

    T1 = "1"
    """The run is identified -- model, harness, dataset -- but the decoding and serving
    conditions are requests rather than read-backs."""

    T2 = "2"
    """Decoding conditions established and the item sample fingerprinted. Two runs at this
    tier can be compared if, and only if, their serving stacks happen to match."""

    T3 = "3"
    """The serving stack is established too: which upstream answered and at what numerical
    precision, read back rather than requested."""


@dataclass
class Field:
    """One condition, its value, and how that value was obtained."""

    name: str
    value: Any
    provenance: Provenance
    note: str = ""
    """Why a field is unavailable or inapplicable, or what a measurement was read from. An
    empty note on a MEASURED field is fine; an empty note on an UNAVAILABLE one is a gap."""

    @property
    def established(self) -> bool:
        return self.provenance in ESTABLISHED and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "provenance": self.provenance.value}
        if self.note:
            out["note"] = self.note
        return out


# The fields each tier requires to be *established*, not merely present. Named here rather
# than inline so the rule can be read without reading the code that applies it.
TIER_REQUIREMENTS: dict[Tier, tuple[str, ...]] = {
    Tier.T1: ("model_id", "harness_name", "harness_version", "dataset_name"),
    Tier.T2: ("temperature", "max_tokens", "epochs", "dataset_fingerprint", "samples_scored"),
    Tier.T3: ("served_provider", "quant_scheme"),
}


@dataclass
class RunConditionsRecord:
    """The conditions that produced a score, each marked with how it was obtained."""

    fields: dict[str, Field] = field(default_factory=dict)
    source: str = ""
    """What this was read from, so a record can be traced back to its evidence."""

    emitter: str = ""
    """Which adapter produced it, with its version. A record is itself a run."""

    def add(self, name: str, value: Any, provenance: Provenance, note: str = "") -> None:
        self.fields[name] = Field(name, value, provenance, note)

    def get(self, name: str) -> Field | None:
        return self.fields.get(name)

    def established(self, name: str) -> bool:
        f = self.fields.get(name)
        return bool(f and f.established)

    @property
    def tier(self) -> Tier:
        """The highest tier whose requirements are *all* established, cumulatively.

        Cumulative on purpose: a record that reads back the serving precision but cannot
        fingerprint its item sample has not earned tier 3, because two runs over different
        items are not comparable however well their hardware is described.
        """
        reached = Tier.T0
        for tier in (Tier.T1, Tier.T2, Tier.T3):
            if not all(self.established(n) for n in TIER_REQUIREMENTS[tier]):
                return reached
            reached = tier
        return reached

    @property
    def missing_for_next_tier(self) -> list[str]:
        """Exactly which fields stand between this record and one tier better.

        A tier with no account of what it costs to improve is a grade, not a diagnosis.
        """
        order = (Tier.T1, Tier.T2, Tier.T3)
        current = self.tier
        if current is Tier.T3:
            return []
        nxt = order[order.index(current) + 1] if current in order else Tier.T1
        return [n for n in TIER_REQUIREMENTS[nxt] if not self.established(n)]

    def counts(self) -> dict[str, int]:
        """How many fields carry each mark. The shape of a record's evidence, at a glance."""
        out: dict[str, int] = {}
        for f in self.fields.values():
            out[f.provenance.value] = out.get(f.provenance.value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rcr_version": "0.1.0",
            "source": self.source,
            "emitter": self.emitter,
            "tier": self.tier.value,
            "missing_for_next_tier": self.missing_for_next_tier,
            "provenance_counts": self.counts(),
            "fields": {k: v.as_dict() for k, v in sorted(self.fields.items())},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)
