"""The note-level representation both sides of an evaluation speak.

A ``NoteEvent`` is the note-level peer of ``Contour``: where a contour is pitch
sampled per frame, a note is a discrete event with a start, an end, and one
pitch. Both are exchanged between datasets (as references) and baselines (as
predictions), so like ``Contour`` this type lives in a neutral module that
neither ``data`` nor ``baselines`` owns -- keeping the dependency direction
one-way and needing no import guards.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteEvent:
    """A detected or ground-truth note."""

    onset: float  # seconds
    offset: float  # seconds
    midi_pitch: float
    confidence: float = 1.0
