"""Baseline pitch estimators and the contract the evaluation harness expects.

A baseline maps one isolated-note clip to a single MIDI pitch. That is all the
IDMT-SMT-Bass stage needs -- every file is exactly one note -- so the contract
is deliberately narrower than full transcription. ``predict_notes`` is sketched
for the continuous-bassline stages (Single-Track, Slakh) but not required here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoteEvent:
    """A detected note, for the future continuous-transcription stages."""

    onset: float  # seconds
    offset: float  # seconds
    midi_pitch: float
    confidence: float = 1.0


class PitchBaseline(ABC):
    """Estimates the pitch of a single recorded note."""

    name: str

    @abstractmethod
    def predict_pitch(self, path: str | Path) -> float | None:
        """MIDI pitch of a single-note clip, or None if nothing voiced is found."""

    def predict_notes(self, path: str | Path) -> list[NoteEvent]:
        """Note events for a continuous clip. Only some baselines implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} only does single-note pitch estimation"
        )
