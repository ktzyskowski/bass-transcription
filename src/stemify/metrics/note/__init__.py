"""Note-level metrics: score lists of per-track note-event lists.

``NoteMetric`` is the contract and ``matched_pairs`` the shared matcher; a note metric
loops tracks and sums counts internally, returning one result (see
``stemify.evaluation.note``).
"""

from stemify.metrics.note.base import (
    NoteMetric,
    count_matches,
    matched_pairs,
)
from stemify.metrics.note.fragmentation import (
    Fragmentation,
    FragmentationResult,
)
from stemify.metrics.note.prf import PRF, PRFResult, prf_rates
from stemify.metrics.note.scoring import (
    DEFAULT_NOTE_METRICS,
    score_notes,
)

__all__ = [
    "NoteMetric",
    "matched_pairs",
    "count_matches",
    "PRF",
    "PRFResult",
    "prf_rates",
    "Fragmentation",
    "FragmentationResult",
    "DEFAULT_NOTE_METRICS",
    "score_notes",
]
