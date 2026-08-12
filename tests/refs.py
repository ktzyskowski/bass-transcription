"""Building a ``Reference`` for tests that have notes but no audio.

``gold_segments`` takes a reference -- notes *and* the mask saying where the bass is
audible -- because those are two views of one truth and deriving the second from the
first is what let gold and the voicing target drift apart. A fixture therefore has to
*supply* a mask, not re-derive one; that is the whole point.

``sounding`` is the simplest such datum: the bass is audible exactly across each note's
MIDI span, plus an optional ring-out. ``ring_s=0`` means a note stops dead at note-off,
which no real bass does but which makes a fixture's intent obvious. Tests that care
about decay pass a ring explicitly, and then they are stating it rather than inheriting
it from a constant.
"""

import numpy as np

from stemify.music.notes import NoteEvent
from stemify.music.reference import Reference
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.semicrf.segmentation import TrackSegmentation


def sounding(
    frame_times: np.ndarray,
    notes: list[NoteEvent],
    ring_s: float = 0.0,
) -> Reference:
    """A reference audible across each note, plus ``ring_s`` capped at the next onset."""
    times = np.asarray(frame_times, dtype=float)
    voiced = np.zeros(times.size, dtype=bool)
    ordered = sorted(notes, key=lambda n: n.onset)
    for k, note in enumerate(ordered):
        nxt = ordered[k + 1].onset if k + 1 < len(ordered) else np.inf
        stop = min(note.offset + ring_s, nxt)
        voiced |= (times >= note.onset) & (times < stop)
    return Reference(frame_times=times, voiced=voiced, notes=tuple(ordered))


def gold(
    frame_times: np.ndarray,
    notes: list[NoteEvent],
    dmax: int,
    labels: LabelSet = BINARY,
    ring_s: float = 0.0,
) -> TrackSegmentation:
    """``gold_segments`` over a ``sounding`` reference, in the old argument order.

    A convenience for the many tests whose subject is the DP rather than the labelling
    rule -- they want *a* gold tiling and do not care how the bass decays.
    ``tests/semicrf/test_gold.py`` deliberately does not use this: it builds references
    by hand, because the mask is exactly what it is testing.
    """
    return gold_segments(sounding(frame_times, notes, ring_s), dmax, labels)
