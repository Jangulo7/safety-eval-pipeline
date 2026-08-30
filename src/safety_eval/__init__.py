"""Safety evaluation pipeline built on AISI Inspect (``inspect_ai`` + ``inspect_evals``).

Runs safety benchmarks across models, compares them honestly, and emits a leaderboard,
charts, a PDF report and a release gate. See ``docs/SPEC.md``.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
