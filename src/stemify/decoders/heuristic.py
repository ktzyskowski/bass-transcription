"""``HeuristicDecoder``: the greedy DSP-era decode (no learning).

Turns per-frame ``Cues`` into notes by splitting the pitch contour at three cues --
a rest (an unvoiced run), a pitch change, and an onset (a same-pitch
re-articulation the contour alone cannot show) -- then rejoining same-pitch
fragments a spurious cue split. The rest/onset cues over-fire (flux triggers on
mid-sustain noise; pyin drops out for a frame), so the optional ``merge_gap_s`` pass
undoes the dominant source of over-segmentation, trading against recall.

The pure core is ``group_contour`` -- no audio, no thresholds, taking onset *times*
and a voiced mask directly -- so a boundary case is testable with a synthetic
contour and hand-picked cues. ``HeuristicDecoder.decode`` is the ``Decoder`` wrapper:
it peak-picks the onset curve and thresholds the voicing curve out of the ``Cues``,
then groups. That thresholding is exactly what a ``SemiCRFDecoder`` does *not* do --
it keeps the soft probabilities -- which is why the two are siblings.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.base import Decoder
from stemify.decoders.frames import resample_mask
from stemify.notes import NoteEvent

# Below this, a "note" is a tracker blip (onset transient, a frame of octave
# wobble) rather than a played note.
MIN_NOTE_S = 0.05
# A sustained jump this far from the note's running pitch starts a new note.
# Under half a semitone would split vibrato; this keeps a vibrato'd note whole.
PITCH_SPLIT_SEMITONES = 0.8
# Two same-pitch notes closer than this are rejoined: the split that separated them
# was an artifact (a spurious mid-sustain onset, gap ~0, or a one-frame pyin dropout,
# gap ~one hop) rather than a real re-articulation. None disables the pass -- the
# over-segmentation/fragmentation lever.
MERGE_GAP_S: float | None = None

# Peak-picking an onset *probability* curve: a local maximum above ONSET_THRESHOLD is
# an onset; MIN_ONSET_GAP_S enforces a refractory gap so one attack does not fire
# twice. VOICING_THRESHOLD cuts the voicing curve into a note/rest mask.
ONSET_THRESHOLD = 0.5
MIN_ONSET_GAP_S = 0.03
VOICING_THRESHOLD = 0.5


def onset_times_from_probs(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = ONSET_THRESHOLD,
    min_gap_s: float = MIN_ONSET_GAP_S,
) -> np.ndarray:
    """Peak-pick an onset-probability curve into onset times.

    A local maximum above ``threshold`` is an onset; ``min_gap_s`` enforces a
    refractory gap so one attack does not fire twice. Pure, so the peak logic is
    tested without a model or audio.
    """
    if probs.size == 0:
        return np.zeros(0)
    hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
    distance = max(1, round(min_gap_s / hop))
    peaks, _ = find_peaks(probs, height=threshold, distance=distance)
    return frame_times[peaks]


def group_contour(
    contour: Contour,
    onsets: np.ndarray | None = None,
    voiced: np.ndarray | None = None,
    *,
    min_note_s: float = MIN_NOTE_S,
    pitch_split_semitones: float = PITCH_SPLIT_SEMITONES,
    merge_gap_s: float | None = MERGE_GAP_S,
) -> list[NoteEvent]:
    """Group a contour into notes given onset times. The pure core -- no audio, no
    I/O -- so a boundary case is testable with a synthetic contour and hand-picked
    onsets.

    ``voiced`` decides where notes sound: by default a frame is voiced iff pyin
    tracked a pitch there (``freqs > 0``), but a learned voicing head can supply its
    own per-frame mask (aligned to ``contour.times``) to place the rests instead.
    Pitch always comes from ``freqs``, so a frame the mask calls voiced but pyin left
    pitchless simply contributes no pitch -- it never invents one.
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
        if offset - onset >= min_note_s and not np.isnan(pitch):
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
            and abs(midi[i] - np.nanmedian(prev)) > pitch_split_semitones
        )
        if onset_frame[i] or jumped:
            emit(start, i)
            start = i
    if start is not None:
        emit(start, times.size)

    if merge_gap_s is not None:
        notes = _merge_same_pitch(notes, merge_gap_s, pitch_split_semitones)
    return notes


def _merge_same_pitch(
    notes: list[NoteEvent], merge_gap_s: float, pitch_split_semitones: float
) -> list[NoteEvent]:
    """Rejoin consecutive notes within ``pitch_split_semitones`` and ``merge_gap_s``
    of each other -- the fragments a spurious onset or a pyin dropout left behind.
    The merged note keeps the first fragment's onset (the real one, already matched
    as a true positive) and drops the second's onset (the phantom), which is exactly
    the false positive to remove. Pitch becomes the duration-weighted mean, so a long
    note plus a stray frame is barely moved."""
    if not notes:
        return notes
    merged = [notes[0]]
    for nxt in notes[1:]:
        cur = merged[-1]
        gap = nxt.onset - cur.offset
        same_pitch = abs(nxt.midi_pitch - cur.midi_pitch) <= pitch_split_semitones
        if same_pitch and gap <= merge_gap_s:
            dur_cur = cur.offset - cur.onset
            dur_nxt = nxt.offset - nxt.onset
            pitch = (
                (cur.midi_pitch * dur_cur + nxt.midi_pitch * dur_nxt)
                / (dur_cur + dur_nxt)
            )
            merged[-1] = NoteEvent(onset=cur.onset, offset=nxt.offset, midi_pitch=pitch)
        else:
            merged.append(nxt)
    return merged


class HeuristicDecoder(Decoder):
    """Greedy decode: peak-pick onsets, threshold voicing, split + merge."""

    name = "heuristic"

    def __init__(
        self,
        min_note_s: float = MIN_NOTE_S,
        pitch_split_semitones: float = PITCH_SPLIT_SEMITONES,
        merge_gap_s: float | None = MERGE_GAP_S,
        onset_threshold: float = ONSET_THRESHOLD,
        min_onset_gap_s: float = MIN_ONSET_GAP_S,
        voicing_threshold: float = VOICING_THRESHOLD,
    ) -> None:
        self.min_note_s = min_note_s
        self.pitch_split_semitones = pitch_split_semitones
        self.merge_gap_s = merge_gap_s
        self.onset_threshold = onset_threshold
        self.min_onset_gap_s = min_onset_gap_s
        self.voicing_threshold = voicing_threshold

    def decode(self, cues: Cues) -> list[NoteEvent]:
        """Peak-pick the onset curve and threshold the voicing curve out of ``cues``,
        align voicing to the contour grid, then group. When the cues carry no voicing
        (``None``), grouping falls back to the contour's own ``freqs > 0``."""
        onsets = onset_times_from_probs(
            cues.onset, cues.frame_times, self.onset_threshold, self.min_onset_gap_s
        )
        voiced = None
        if cues.voicing is not None:
            mask = cues.voicing > self.voicing_threshold
            voiced = resample_mask(cues.frame_times, mask, cues.contour.times).astype(bool)
        return group_contour(
            cues.contour,
            onsets,
            voiced,
            min_note_s=self.min_note_s,
            pitch_split_semitones=self.pitch_split_semitones,
            merge_gap_s=self.merge_gap_s,
        )
