"""``HeuristicSegmenter``: the DSP floor (no learning).

A baseline to beat before a note-level model is worth building, exactly as pyin is
the floor for the pitch stage. It draws a note boundary at any of three cues:

  - **a rest** -- an unvoiced run separates two notes (note -> silence -> note)
  - **a pitch change** -- the contour jumps to a different pitch
  - **an onset** -- a re-articulation at the *same* pitch, which the contour alone
    cannot show. Two repeated quarter-notes and one held half-note produce an
    identical contour; only an onset distinguishes them. Onsets come from spectral
    flux on the audio; without them repeated same-pitch notes merge into one.

The grouping itself -- split at those cues, then the ``merge_gap_s`` repair pass --
now lives in ``stemify.decoders.heuristic`` (``group_contour``); this segmenter
supplies the DSP onset cue (spectral flux) and delegates. It is the pre-refactor
``Segmenter`` API kept working; the new pipeline expresses the same thing as
``Segmenter(DSPEncoder(), HeuristicDecoder())``.
"""

from pathlib import Path

import numpy as np

from stemify.contour import Contour
from stemify.decoders.heuristic import (
    MERGE_GAP_S,
    MIN_NOTE_S,
    PITCH_SPLIT_SEMITONES,
    group_contour,
)
from stemify.encoders.dsp import ONSET_DELTA, detect_onsets
from stemify.notes import NoteEvent
from stemify.segmentation.base import Segmenter


class HeuristicSegmenter(Segmenter):
    """The DSP floor: split at rests, pitch changes, and detected onsets.

    ``segment`` is the public entry point (detects onsets, then groups); ``group``
    delegates to the shared ``group_contour`` core, taking onsets as an argument so
    it can be exercised without audio.
    """

    name = "heuristic"

    def __init__(
        self,
        min_note_s: float = MIN_NOTE_S,
        pitch_split_semitones: float = PITCH_SPLIT_SEMITONES,
        use_onsets: bool = True,
        merge_gap_s: float | None = MERGE_GAP_S,
        onset_delta: float | None = ONSET_DELTA,
    ) -> None:
        self.min_note_s = min_note_s
        self.pitch_split_semitones = pitch_split_semitones
        # Onsets are the only cue for a same-pitch re-articulation. Off, those
        # notes merge -- useful to toggle when measuring how much onset
        # detection contributes to note recall.
        self.use_onsets = use_onsets
        # Rejoin same-pitch notes split by an artifact rather than a real
        # re-articulation. None keeps the raw splits; a small value undoes the
        # dominant source of over-segmentation.
        self.merge_gap_s = merge_gap_s
        # Onset peak-picking threshold. Higher drops weak (spurious) onsets --
        # the better-targeted fragmentation lever, since it removes only weak
        # splits rather than merging indiscriminately. None = librosa default.
        self.onset_delta = onset_delta

    def segment(self, contour: Contour, audio_path: str | Path) -> list[NoteEvent]:
        return self.group(contour, self._onset_times(audio_path))

    def group(
        self,
        contour: Contour,
        onsets: np.ndarray | None = None,
        voiced: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Group a contour into notes given onset times -- delegates to the shared
        ``group_contour`` core in ``stemify.decoders.heuristic``."""
        return group_contour(
            contour,
            onsets,
            voiced,
            min_note_s=self.min_note_s,
            pitch_split_semitones=self.pitch_split_semitones,
            merge_gap_s=self.merge_gap_s,
        )

    def _onset_times(self, audio_path: str | Path) -> np.ndarray | None:
        """Onset times (seconds) used as the same-pitch split cue -- here, spectral
        flux. The one piece a learned segmenter swaps: override this to peak-pick a
        trained onset head instead, and the grouping (rests, pitch changes, merge)
        is inherited unchanged."""
        if not self.use_onsets:
            return None
        return detect_onsets(audio_path, delta=self.onset_delta)
