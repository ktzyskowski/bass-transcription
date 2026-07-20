"""Baseline pitch estimators and the contract the evaluation harness expects.

A baseline answers one of two ways. ``predict_f0`` gives a frame-level contour
and is what the frame-level trackers do naturally. ``predict_pitch`` gives a
single pitch for a clip, which is all an isolated-note dataset needs and all
some off-the-shelf transcribers expose.

Either is enough to be evaluated: a single pitch is a degenerate contour holding
one value across the clip (see ``Contour.constant``), so both kinds of baseline
are scored by the same metrics rather than needing parallel evaluation paths.

``predict_notes`` is sketched for the note-level stages -- once segmentation
exists, transcription needs onsets and offsets, not just pitch -- but nothing
requires it yet.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from stemify.contour import Contour
from stemify.notes import NoteEvent


class PitchBaseline(ABC):
    """Estimates the pitch of recorded audio."""

    name: str

    @abstractmethod
    def predict_pitch(self, path: str | Path) -> float | None:
        """MIDI pitch of a clip, or None if nothing voiced is found."""

    def predict_f0(self, path: str | Path) -> Contour:
        """Frame-level pitch track. Implemented by the frame-level estimators."""
        raise NotImplementedError(
            f"{type(self).__name__} does not produce a frame-level f0 contour"
        )

    @property
    def is_frame_level(self) -> bool:
        """Whether ``predict_f0`` is real, or inherited and raising.

        The evaluation layer asks this to decide whether it must synthesise a
        contour from ``predict_pitch`` instead of calling ``predict_f0``.
        """
        return type(self).predict_f0 is not PitchBaseline.predict_f0

    def predict_notes(self, path: str | Path) -> list[NoteEvent]:
        """Note events for a continuous clip. Only some baselines implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not produce note events"
        )
