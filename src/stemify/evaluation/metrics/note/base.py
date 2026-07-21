"""``NoteMetric``: the contract for a note-level metric, and the shared matcher.

A ``NoteMetric`` scores a *dataset* of note events -- reference and estimate note
lists, one pair per track -- and returns a single ``MetricResult``. It takes the
whole dataset at once, exactly like ``FrameMetric`` takes the whole pooled array
at once, so both families have the same one-method shape: score everything,
return the final numbers.

The difference is only in *why the input has the shape it does*:

  - A frame metric receives ``(true, pred)`` already flattened across tracks,
    because frame scoring is elementwise -- frame *i* compares to frame *i* and
    to nothing else -- so concatenation is lossless.
  - A note metric receives ``(references, estimates)`` as *lists of per-track
    lists*, because note matching is a per-track set assignment: notes compete
    to match, and a note in one track must never match one in another. The track
    boundary is load-bearing, so it cannot be flattened away. The metric loops
    tracks, matches each, and sums the counts into one result.

That internal sum is the whole reason the input is nested rather than flat, and
it is why rates (precision, F1) are derived once at the end from pooled counts,
never averaged per track (per-track F1 is not averageable). A per-track number
-- for ranking the worst items -- is the same ``calculate`` on a one-track list.

``matched_pairs`` is defined here, once, and shared by every note metric (``prf``,
``fragmentation``), so P/R/F and the fragmentation breakdown can never
disagree about which estimates were matched. It ignores offsets
(``offset_ratio=None``): a bass note's end is fuzzy (decay, release), its start
is not, so onset + pitch is the honest criterion. Onset tolerance defaults to
50 ms and pitch tolerance to 50 cents, the ``mir_eval`` conventions, matching the
frame-level protocol.
"""

from abc import ABC, abstractmethod

import mir_eval
import numpy as np

from stemify.evaluation.metrics.base import MetricResult
from stemify.notes import NoteEvent

ONSET_TOLERANCE_S = 0.05
PITCH_TOLERANCE_CENTS = 50.0


def _to_arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    """(intervals Nx2 seconds, pitches N in Hz), the mir_eval note format."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.onset, n.offset] for n in notes])
    pitches = 440.0 * 2.0 ** ((np.array([n.midi_pitch for n in notes]) - 69.0) / 12.0)
    return intervals, pitches


def matched_pairs(
    ref: list[NoteEvent],
    est: list[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE_S,
    pitch_tolerance: float = PITCH_TOLERANCE_CENTS,
) -> list[tuple[int, int]]:
    """(reference index, estimate index) pairs for one track -- the shared matcher.

    Defining "the same note" exactly once -- onset within tolerance, pitch
    within tolerance, offsets ignored -- means every note-level number is built
    on the identical matching. An empty side yields no pairs;
    ``mir_eval.match_notes`` itself expects non-empty inputs, so that is handled
    here.
    """
    if not ref or not est:
        return []
    ref_int, ref_hz = _to_arrays(ref)
    est_int, est_hz = _to_arrays(est)
    return mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,  # onset + pitch only; bass offsets are fuzzy
    )


def count_matches(
    ref: list[NoteEvent],
    est: list[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE_S,
    pitch_tolerance: float = PITCH_TOLERANCE_CENTS,
) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one track.

    An estimated note is a true positive if some reference note matches it in
    onset and pitch; each reference note matches at most one estimate. Everything
    unmatched on the estimate side is a false positive, on the reference side a
    false negative.
    """
    tp = len(matched_pairs(ref, est, onset_tolerance, pitch_tolerance))
    return tp, len(est) - tp, len(ref) - tp


class NoteMetric(ABC):
    """Scores a dataset of (reference, estimate) note lists into one result.

    One abstract method, ``calculate``, taking the whole dataset as parallel
    lists of per-track note lists (see the module docstring for why the input is
    nested rather than flat). Both tolerances are held on the instance so every
    note metric in a run agrees on what "the same note" means.
    """

    name: str

    def __init__(
        self,
        onset_tolerance: float = ONSET_TOLERANCE_S,
        pitch_tolerance: float = PITCH_TOLERANCE_CENTS,
    ) -> None:
        self.onset_tolerance = onset_tolerance
        self.pitch_tolerance = pitch_tolerance

    @abstractmethod
    def calculate(
        self,
        references: list[list[NoteEvent]],
        estimates: list[list[NoteEvent]],
    ) -> MetricResult:
        """Score every track (matching per track, counts summed) into one result.

        ``references`` and ``estimates`` are parallel: track *i*'s reference and
        estimate note lists. A single-element pair scores one track, which is how
        the harness gets per-item numbers for the worst-item breakdown.
        """