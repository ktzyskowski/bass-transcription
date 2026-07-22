"""The segmenter interface: frame-level pitch contour -> discrete note events.

Segmentation is the step from "what pitch at each instant" to "which notes were
played" -- a first-class, swappable component, not a method hidden on the pitch
tracker, because it is the part of the note-level pipeline this project studies. A
note-level score is ``pitch-tracker + segmenter``, and the segmenter is the weak,
interesting half (onset and voicing). Keeping the interface separate is what lets
the pitch tracker stay fixed while segmenters are compared, and lets a learned
onset/voicing head drop in as just another ``Segmenter``.

Monophonic bass is what makes this tractable: with one note at a time the timeline
partitions cleanly into notes, no overlapping voices to untangle.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from stemify.contour import Contour
from stemify.notes import NoteEvent


class Segmenter(ABC):
    """Turns a pitch contour (and its audio) into note events."""

    name: str

    @abstractmethod
    def segment(self, contour: Contour, audio_path: str | Path) -> list[NoteEvent]:
        """The notes implied by ``contour``. ``audio_path`` is available for
        cues the contour does not carry, chiefly onsets."""
