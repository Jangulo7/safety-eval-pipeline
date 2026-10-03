"""rcr-core: a Run Conditions Record, emitted from an evaluation log.

A benchmark score is not a property of a model. It is a property of a *run*: the model, the
harness, the judge, the item sample, the decoding parameters and the serving stack together.
This library reads an evaluation log and emits the conditions that produced the score, with
one mark per field saying **how that value was obtained**.

The mark is the point
---------------------
Most conditions reporting states what was configured. That describes an intention. A parameter
sent to a provider that discarded it, and a parameter read back from the run that applied it,
look identical once printed -- and the difference is the whole question of whether a number can
be compared with another number. So every field carries a :class:`Provenance`, and a record
that cannot establish a field says so rather than restating the request.

Harness-neutral by construction
-------------------------------
Nothing here imports the pipeline that happens to live beside it. The core defines the record;
adapters map one harness's log onto it. That separation is load-bearing: a record format that
only works on logs produced by its own author's tooling cannot make claims about anyone else's.
"""

from .record import (
    Field,
    Provenance,
    RunConditionsRecord,
    Tier,
)

__all__ = ["Field", "Provenance", "RunConditionsRecord", "Tier"]
__version__ = "0.1.0"
