"""Visual constants shared by the PNG charts, the HTML leaderboard and the PDF.

The categorical hues are a validated three-slot palette. Three is not an arbitrary cap: the
scatter and the small-multiples grid put every series against every other one, and beyond
three slots the all-pairs colour-vision separation floor cannot be cleared. With more models
than that, identity moves to facets and direct labels rather than to a fourth hue.

Aqua sits below 3:1 against the light surface, so every chart that uses it carries visible
direct labels — identity and value are never carried by colour alone.
"""

from __future__ import annotations

# Categorical slots, in fixed order. Colour follows the entity, never its rank, so a model
# keeps its hue when the table is re-sorted or filtered.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_OVERFLOW = "#898781"  # a 4th+ entity folds to a neutral and relies on its label

# Reserved for state, never for a series.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "#e1e0d9"

FONT_STACK = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
DPI = 200


def series_color(index: int) -> str:
    """Hue for the n-th entity, folding to a neutral past the validated slot count."""
    return SERIES[index] if index < len(SERIES) else SERIES_OVERFLOW
