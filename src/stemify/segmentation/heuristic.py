"""``HeuristicSegmenter``: the DSP floor (no learning).

A baseline to beat before a note-level model is worth building, exactly as pyin is
the floor for the pitch stage. It draws a note boundary at any of three cues:

  - **a rest** -- an unvoiced run separates two notes (note -> silence -> note)
  - **a pitch change** -- the contour jumps to a different pitch
  - **an onset** -- a re-articulation at the *same* pitch, which the contour alone
    cannot show. Two repeated quarter-notes and one held half-note produce an
    identical contour; only an onset distinguishes them. Onsets come from spectral
    flux on the audio; without them repeated same-pitch notes merge into one.

The rest/onset cues over-fire (flux triggers on mid-sustain noise; pyin drops out
for a frame or two), both splitting one held note into same-pitch fragments. So an
optional final pass (``merge_gap_s``) rejoins consecutive same-pitch notes closer
than a small gap. It trades against recall (a real fast repeat can look like a
fragment), which is exactly the knee the Fragmentation metric measures.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.segmentation.base import Segmenter

# Below this, a "note" is a tracker blip (onset transient, a frame of octave
# wobble) rather than a played note.
MIN_NOTE_S = 0.05
# A sustained jump this far from the note's running pitch starts a new note.
# Under half a semitone would split vibrato; this keeps a vibrato'd note whole.
PITCH_SPLIT_SEMITONES = 0.8
# Two same-pitch notes closer than this are rejoined: the split that separated
# them was an artifact (a spurious mid-sustain onset, gap ~0, or a one-frame pyin
# dropout, gap ~one hop) rather than a real re-articulation. None disables the
# pass. This is the over-segmentation/fragmentation lever: too large and genuine
# repeated same-pitch notes merge, costing recall. Tuned against the
# Fragmentation metric.
MERGE_GAP_S: float | None = None
# Peak-picking threshold for onset detection: an onset must rise this far above
# the local mean flux to fire. librosa's default is 0.07. Raising it drops the
# weak flux peaks -- mid-sustain bow/finger noise -- that spuriously split a held
# note, while keeping the strong attacks of real re-plucks, because spurious
# onsets are systematically weaker than real ones. A sweep on Slakh stems put the
# F1 optimum near 0.20 (F1 71% -> 73%, precision +7 pts at -6 pts recall), which
# strictly beats the merge pass. Left at None (librosa default) because that
# optimum is tuned on isolated stems; the separated/mixed deployment condition
# has a different flux distribution and needs its own validation before the
# default moves. Unlike a scalar threshold, a learned onset head sees the whole
# attack shape, so this is the ceiling it should beat.
ONSET_DELTA: float | None = None


def detect_onsets(
    path: str | Path, sr: int = 22050, delta: float | None = ONSET_DELTA
) -> np.ndarray:
    """Onset times (seconds) via spectral flux, backtracked to the energy rise.

    A module-level utility rather than a ``Segmenter`` method because it is a
    generic audio operation: a learned segmenter might reuse it, replace it with
    a learned onset head, or ignore it. ``backtrack`` moves each detection to the
    preceding local energy minimum, so the reported time is the note's actual
    start rather than the flux peak a few frames later. ``delta`` is the
    peak-picking threshold; None uses librosa's default (0.07).
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    kwargs = {} if delta is None else {"delta": delta}
    return librosa.onset.onset_detect(
        y=audio, sr=sr, units="time", backtrack=True, **kwargs
    )


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

    def _onset_times(self, audio_path: str | Path) -> np.ndarray | None:
        """Onset times (seconds) used as the same-pitch split cue -- here, spectral
        flux. The one piece a learned segmenter swaps: override this to peak-pick a
        trained onset head instead, and the grouping (rests, pitch changes, merge)
        is inherited unchanged."""
        if not self.use_onsets:
            return None
        return detect_onsets(audio_path, delta=self.onset_delta)

    def group(
        self,
        contour: Contour,
        onsets: np.ndarray | None = None,
        voiced: np.ndarray | None = None,
    ) -> list[NoteEvent]:
        """Group a contour into notes given onset times. The pure core -- no
        audio, no I/O -- so a boundary case is testable with a synthetic contour
        and hand-picked onsets.

        ``voiced`` decides where notes sound: by default a frame is voiced iff pyin
        tracked a pitch there (``freqs > 0``), but a learned voicing head can supply
        its own per-frame mask (aligned to ``contour.times``) to place the rests
        instead. Pitch always comes from ``freqs``, so a frame the mask calls voiced
        but pyin left pitchless simply contributes no pitch -- it never invents one.
        """
        times, freqs = contour.times, contour.freqs
        if times.size == 0:
            return []

        has_pitch = freqs > 0
        midi = np.full(freqs.shape, np.nan)
        midi[has_pitch] = librosa.hz_to_midi(freqs[has_pitch])
        if voiced is None:
            voiced = has_pitch
        else:
            voiced = np.asarray(voiced, dtype=bool)
            if voiced.shape != times.shape:
                raise ValueError(
                    f"voiced mask {voiced.shape} must match contour grid {times.shape}"
                )

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
            window = midi[lo:hi]
            if np.isnan(window).all():  # learned-voiced but pyin tracked no pitch
                return
            pitch = float(np.nanmedian(window))
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
            # Voiced continuation: does a boundary fall here? (No pitch to compare
            # against when the run so far is all pyin-pitchless -> no pitch split.)
            prev = midi[start:i]
            jumped = (
                not np.isnan(prev).all()
                and abs(midi[i] - np.nanmedian(prev)) > self.pitch_split_semitones
            )
            if onset_frame[i] or jumped:
                emit(start, i)
                start = i
        if start is not None:
            emit(start, times.size)

        if self.merge_gap_s is not None:
            notes = self._merge_same_pitch(notes)
        return notes

    def _merge_same_pitch(self, notes: list[NoteEvent]) -> list[NoteEvent]:
        """Rejoin consecutive notes within ``pitch_split_semitones`` and
        ``merge_gap_s`` of each other -- the fragments a spurious onset or a pyin
        dropout left behind. The merged note keeps the first fragment's onset
        (the real one, already matched as a true positive) and drops the second's
        onset (the phantom), which is exactly the false positive to remove. Pitch
        becomes the duration-weighted mean, so a long note plus a stray frame is
        barely moved."""
        if not notes:
            return notes
        merged = [notes[0]]
        for nxt in notes[1:]:
            cur = merged[-1]
            gap = nxt.onset - cur.offset
            same_pitch = abs(nxt.midi_pitch - cur.midi_pitch) <= self.pitch_split_semitones
            if same_pitch and gap <= self.merge_gap_s:
                dur_cur = cur.offset - cur.onset
                dur_nxt = nxt.offset - nxt.onset
                pitch = (
                    (cur.midi_pitch * dur_cur + nxt.midi_pitch * dur_nxt)
                    / (dur_cur + dur_nxt)
                )
                merged[-1] = NoteEvent(
                    onset=cur.onset, offset=nxt.offset, midi_pitch=pitch
                )
            else:
                merged.append(nxt)
        return merged
