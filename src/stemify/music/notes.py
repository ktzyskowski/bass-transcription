"""The note-level representation both sides of an evaluation speak.

A ``NoteEvent`` is the note-level peer of ``Contour``: where a contour is pitch
sampled per frame, a note is a discrete event with a start, an end, and one
pitch. Both are exchanged between datasets (as references) and the pipeline (as
predictions), so like ``Contour`` this type lives in a neutral module that neither
``data`` nor ``decoders`` owns -- keeping the dependency direction one-way and
needing no import guards.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteEvent:
    """A detected or ground-truth note."""

    onset: float  # seconds
    offset: float  # seconds
    midi_pitch: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        # Checked rather than assumed because the two fields are the same kind of number
        # in the same unit, so a swap type-checks, reads plausibly, and survives to the
        # metric -- where ``mir_eval.match_notes`` scores on onsets alone (offsets are
        # ignored, see ``metrics.note.base``) and reports a perfectly ordinary F1. The
        # decoders build spans from an ordered frame range and cannot produce this; a
        # reference adapter reading a corpus's own fields can.
        if not self.offset >= self.onset:
            raise ValueError(
                f"note ends before it starts: onset={self.onset}, "
                f"offset={self.offset}. Both are seconds on the same clock."
            )
