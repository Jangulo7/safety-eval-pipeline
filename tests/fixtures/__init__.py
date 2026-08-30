"""Synthetic fixtures for offline tests.

**These numbers are invented.** They exist so that renderers, gates and the leaderboard can
be exercised without provider access, and they must never reach a published artefact. Every
factory here stamps ``notes`` on the run metadata saying so, and ``test_reporting`` asserts
that a published README never cites a run whose notes carry the marker.
"""

from .factory import SYNTHETIC_MARKER, make_results  # noqa: F401
