"""Note-level metrics: score lists of per-track note-event lists.

``NoteMetric`` is the contract and ``matched_pairs`` the shared matcher; a note metric
loops tracks and sums counts internally, returning one result (see
``stemify.evaluation.note``).
"""

from stemify.evaluation.metrics.note.base import (
    NoteMetric,
    count_matches,
    matched_pairs,
)
from stemify.evaluation.metrics.note.fragmentation import (
    Fragmentation,
    FragmentationResult,
)
from stemify.evaluation.metrics.note.prf import PRF, PRFResult

__all__ = [
    "NoteMetric",
    "matched_pairs",
    "count_matches",
    "PRF",
    "PRFResult",
    "Fragmentation",
    "FragmentationResult",
]