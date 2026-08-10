"""Note-level metrics: score lists of per-track note-event lists.

``NoteMetric`` is the contract, ``matched_pairs`` the shared matcher and ``Counts``
the unit both the pooled headline and every per-track row are expressed in; a note
metric loops tracks and sums counts internally, returning one result (see
``stemify.evaluation.note``).
"""

from stemify.metrics.note.base import (
    Counts,
    NoteMetric,
    count_matches,
    matched_pairs,
)
from stemify.metrics.note.fragmentation import (
    Fragmentation,
    FragmentationResult,
)
from stemify.metrics.note.prf import PRF, PRFResult
from stemify.metrics.note.scoring import (
    DEFAULT_NOTE_METRICS,
    score_notes,
)

__all__ = [
    "NoteMetric",
    "Counts",
    "matched_pairs",
    "count_matches",
    "PRF",
    "PRFResult",
    "Fragmentation",
    "FragmentationResult",
    "DEFAULT_NOTE_METRICS",
    "score_notes",
]
