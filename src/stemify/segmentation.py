"""Frame-level pitch contour -> discrete note events.

This is the segmentation stage: the step from "what pitch at each instant" to
"which notes were played". It is a first-class, swappable component -- not a
method hidden on the pitch tracker -- because it is the part of the note-level
pipeline this project actually studies. A note-level score is
``pitch-tracker + segmenter``, and the segmenter is the weak, interesting half
(onset and voicing, which the frame-level work identified as the bottleneck).
Keeping it separate is what lets the pitch tracker stay fixed while segmenters
are compared, and lets a learned onset/voicing head later drop in as just
another ``Segmenter`` -- with its own algorithm, sharing none of the heuristic
below.

``HeuristicSegmenter`` is the DSP floor -- no learning -- so there is a baseline
to beat before a note-level model is worth building, exactly as pyin is the
floor for the pitch stage. It draws a note boundary at any of three cues:

  - **a rest** -- an unvoiced run separates two notes (note -> silence -> note)
  - **a pitch change** -- the contour jumps to a different pitch
  - **an onset** -- a re-articulation at the *same* pitch, which the contour
    alone cannot show. Two repeated quarter-notes and one held half-note produce
    an identical contour; only an onset distinguishes them. Onsets come from
    spectral flux on the audio, and without them repeated same-pitch notes merge
    into one -- a real limitation the note-level metric will expose as missed
    notes.

Monophonic bass is what makes this tractable: with one note at a time the
timeline partitions cleanly into notes, no overlapping voices to untangle.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.notes import NoteEvent

# Below this, a "note" is a tracker blip (onset transient, a frame of octave
# wobble) rather than a played note.
MIN_NOTE_S = 0.05
# A sustained jump this far from the note's running pitch starts a new note.
# Under half a semitone would split vibrato; this keeps a vibrato'd note whole.
PITCH_SPLIT_SEMITONES = 0.8


def detect_onsets(path: str | Path, sr: int = 22050) -> np.ndarray:
    """Onset times (seconds) via spectral flux, backtracked to the energy rise.

    A module-level utility rather than a ``Segmenter`` method because it is a
    generic audio operation: a learned segmenter might reuse it, replace it with
    a learned onset head, or ignore it. ``backtrack`` moves each detection to the
    preceding local energy minimum, so the reported time is the note's actual
    start rather than the flux peak a few frames later.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return librosa.onset.onset_detect(y=audio, sr=sr, units="time", backtrack=True)


class Segmenter(ABC):
    """Turns a pitch contour (and its audio) into note events."""

    name: str

    @abstractmethod
    def segment(self, contour: Contour, audio_path: str | Path) -> list[NoteEvent]:
        """The notes implied by ``contour``. ``audio_path`` is available for
        cues the contour does not carry, chiefly onsets."""


class HeuristicSegmenter(Segmenter):
    """The DSP floor: split at rests, pitch changes, and detected onsets.

    Owns its whole algorithm. ``segment`` is the public entry point (detects
    onsets, then groups); ``group`` is the pure contour-to-notes core, taking
    onsets as an argument so it can be exercised without audio.
    """

    name = "heuristic"

    def __init__(
        self,
        min_note_s: float = MIN_NOTE_S,
        pitch_split_semitones: float = PITCH_SPLIT_SEMITONES,
        use_onsets: bool = True,
    ) -> None:
        self.min_note_s = min_note_s
        self.pitch_split_semitones = pitch_split_semitones
        # Onsets are the only cue for a same-pitch re-articulation. Off, those
        # notes merge -- useful to toggle when measuring how much onset
        # detection contributes to note recall.
        self.use_onsets = use_onsets

    def segment(self, contour: Contour, audio_path: str | Path) -> list[NoteEvent]:
        onsets = detect_onsets(audio_path) if self.use_onsets else None
        return self.group(contour, onsets)

    def group(
        self, contour: Contour, onsets: np.ndarray | None = None
    ) -> list[NoteEvent]:
        """Group a contour into notes given onset times. The pure core -- no
        audio, no I/O -- so a boundary case is testable with a synthetic contour
        and hand-picked onsets."""
        times, freqs = contour.times, contour.freqs
        if times.size == 0:
            return []

        voiced = freqs > 0
        midi = np.full(freqs.shape, np.nan)
        midi[voiced] = librosa.hz_to_midi(freqs[voiced])

        # Mark frames where an onset lands, so a same-pitch re-articulation splits.
        onset_frame = np.zeros(times.shape, dtype=bool)
        if onsets is not None:
            for t in onsets:
                j = int(np.searchsorted(times, t))
                if 0 <= j < times.size:
                    onset_frame[j] = True

        hop = float(np.median(np.diff(times))) if times.size > 1 else 0.01
        notes: list[NoteEvent] = []
        start: int | None = None  # index of the current note's first frame

        def emit(lo: int, hi: int) -> None:
            pitch = float(np.nanmedian(midi[lo:hi]))
            onset, offset = float(times[lo]), float(times[hi - 1] + hop)
            if offset - onset >= self.min_note_s and not np.isnan(pitch):
                notes.append(NoteEvent(onset=onset, offset=offset, midi_pitch=pitch))

        for i in range(times.size):
            if not voiced[i]:
                if start is not None:
                    emit(start, i)
                    start = None
                continue
            if start is None:
                start = i
                continue
            # Voiced continuation: does a boundary fall here?
            jumped = (
                abs(midi[i] - np.nanmedian(midi[start:i]))
                > self.pitch_split_semitones
            )
            if onset_frame[i] or jumped:
                emit(start, i)
                start = i
        if start is not None:
            emit(start, times.size)

        return notes
