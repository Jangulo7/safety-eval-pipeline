"""Rendering of results into artefacts: charts, markdown, HTML leaderboard, PDF report.

Everything in this package renders from ``results/<run>/results.json`` and the benchmark
catalog, never from a live Inspect log. That means every published artefact can be
regenerated and audited without provider access, and it makes the "never invent results"
rule mechanically enforceable: there is no path from a renderer to a number that did not
come out of a committed record.
"""

from .charts import ChartSet, render_charts  # noqa: F401
from .markdown import render_results_markdown  # noqa: F401
