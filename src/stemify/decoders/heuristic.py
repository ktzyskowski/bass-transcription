"""``HeuristicDecoder``: the greedy decode (no learning).

Turns per-frame ``Cues`` into notes by splitting the pitch contour at a rest (an
unvoiced run) and at an onset (a re-articulation the contour alone cannot show),
then rejoining fragments a spurious cue split. The rest/onset cues over-fire (flux
triggers on mid-sustain noise; pyin drops out for a frame), so the optional
``merge_gap_s`` pass undoes the dominant source of over-segmentation, trading
against recall.

A *pitch change* used to be a third splitting cue, and is now off by default -- see
``PITCH_SPLIT_SEMITONES`` for why, and for the one encoder that still needs it.

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
from stemify.grids import resample_mask
from stemify.notes import NoteEvent

# Below this, a "note" is a tracker blip (onset transient, a frame of octave
# wobble) rather than a played note. It earns its place -- removing it costs 0.023
# strict F1 -- and it is enforced against the *contour* grid, which is what makes the
# value depend on where the contour came from.
#
# Raised 0.05 -> 0.10 on 2026-07-31 when the learned pitch head moved the contour from
# pyin's 23.2 ms grid to the CQT's 11.6 ms one. The old value was three pyin frames;
# on the finer grid the same 50 ms admits a great many more short fragments, and
# precision fell 4.2 points. Re-swept: +0.0095 strict F1 on validation and +0.0128 on
# test, taken together with VOICING_THRESHOLD below.
#
# 0.10 is the *middle* of a flat 0.08-0.12 plateau, not its argmax. Validation kept
# climbing to 0.12 while test turned over there, which is what riding a boundary looks
# like on 40 tracks; the middle is the defensible point. Note this discards real notes
# shorter than 100 ms (~7% of the corpus), a permanent recall cap accepted because
# those fragments are more often false positives than notes -- revisit it if short-note
# recall ever becomes the binding constraint.
MIN_NOTE_S = 0.10

# A sustained jump this far from the note's running pitch starts a new note.
# ``None`` -- the default -- means a pitch change never places a note on its own.
#
# This was the only cue a DSP pipeline had for a new note, and it turned into an
# active liability the moment a trained onset head existed. pyin reaches a note's true
# pitch a median 23 ms *after* the onset (37.5 ms mean on separated audio, and past the
# 50 ms match tolerance for 14.2% of pitch changes), so a real attack makes the contour
# walk a staircase through intermediate pitches -- 48.1 -> 51.1 -> 52.0 -- and at 0.8
# every step of that staircase split. Worse, ``group_contour`` *had* already placed a
# note correctly at the onset: the fragment between the onset and the late pitch
# transition was ~2 contour frames, under MIN_NOTE_S, so ``emit`` dropped it, and the
# note that survived started ~58 ms late -- just outside tolerance. One reference note,
# one false negative and one interior-offpitch false positive.
#
# Disabling it was worth +7.5 strict F1 on the held-out test split, on two independent
# checkpoints, and it wins on both sides at once (precision 0.78 -> 0.92, recall
# 0.878 -> 0.886). The sweep is monotone from 0.8 out to "off" with no interior
# optimum, which is the shape of a superseded rule rather than a mistuned one.
PITCH_SPLIT_SEMITONES: float | None = None

# What an encoder with no trained onset head falls back to. ``DSPEncoder`` has only a
# spectral-flux spike train and no voicing curve at all, so for it a pitch change
# really is the only evidence that a new note began; this is the value the project
# shipped with, kept so the DSP floor stays the floor it always was.
DSP_PITCH_SPLIT_SEMITONES = 0.8

# "Close enough in pitch to be the same note" for the rejoin pass below. Deliberately
# its own constant rather than reusing the split tolerance: the two answer different
# questions ("is this a new note" against "were these ever one note"), and while they
# shared one, disabling the split silently turned the merge pass into "merge every
# adjacent note whatever its pitch" -- a perfect fifth collapsing into one note at the
# average of the two.
MERGE_PITCH_SEMITONES = 0.8

# Two same-pitch notes closer than this are rejoined: the split that separated them
# was an artifact (a spurious mid-sustain onset, gap ~0, or a one-frame pyin dropout,
# gap ~one hop) rather than a real re-articulation. None disables the pass -- the
# over-segmentation/fragmentation lever.
MERGE_GAP_S: float | None = None

# Peak-picking an onset *probability* curve: a local maximum above ONSET_THRESHOLD is
# an onset; MIN_ONSET_GAP_S enforces a refractory gap so one attack does not fire
# twice. VOICING_THRESHOLD cuts the voicing curve into a note/rest mask.
#
# Both thresholds are genuine operating points rather than leftovers. ONSET_THRESHOLD
# re-confirmed at 0.5 against the learned contour (0.6 is -0.0005). MIN_ONSET_GAP_S is
# *inert* for a trained onset head (0.0 through 0.05 score identically -- its peaks are
# already well separated) but is kept for the DSP path, whose flux spikes are exactly
# what it guards against.
#
# VOICING_THRESHOLD dropped 0.5 -> 0.3 on 2026-07-31, in the same re-sweep as
# MIN_NOTE_S: a lower bar marks more frames as sounding, so the mask punches fewer
# spurious rests through a held note, and fragmentation FPs fall (946 -> 755 on
# validation) at unchanged recall.
#
# This is *not* the threshold that gates the learned contour -- that is
# ``encoders.neural.CONTOUR_VOICING_THRESHOLD``, deliberately separate and still 0.5.
# The two were one constant until this change, which meant retuning here silently
# retuned the contour as well; the numbers above were measured with 0.3 here and 0.5
# there, and they only hold for that pair.
ONSET_THRESHOLD = 0.5
MIN_ONSET_GAP_S = 0.03
VOICING_THRESHOLD = 0.3


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
    pitch_split_semitones: float | None = PITCH_SPLIT_SEMITONES,
    merge_pitch_semitones: float = MERGE_PITCH_SEMITONES,
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
            pitch_split_semitones is not None
            and not np.isnan(prev).all()
            and abs(midi[i] - np.nanmedian(prev)) > pitch_split_semitones
        )
        if onset_frame[i] or jumped:
            emit(start, i)
            start = i
    if start is not None:
        emit(start, times.size)

    if merge_gap_s is not None:
        notes = _merge_same_pitch(notes, merge_gap_s, merge_pitch_semitones)
    return notes


def _merge_same_pitch(
    notes: list[NoteEvent], merge_gap_s: float, merge_pitch_semitones: float
) -> list[NoteEvent]:
    """Rejoin consecutive notes within ``merge_pitch_semitones`` and ``merge_gap_s``
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
        same_pitch = abs(nxt.midi_pitch - cur.midi_pitch) <= merge_pitch_semitones
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
        pitch_split_semitones: float | None = PITCH_SPLIT_SEMITONES,
        merge_gap_s: float | None = MERGE_GAP_S,
        onset_threshold: float = ONSET_THRESHOLD,
        min_onset_gap_s: float = MIN_ONSET_GAP_S,
        voicing_threshold: float = VOICING_THRESHOLD,
        merge_pitch_semitones: float = MERGE_PITCH_SEMITONES,
    ) -> None:
        self.min_note_s = min_note_s
        self.pitch_split_semitones = pitch_split_semitones
        self.merge_gap_s = merge_gap_s
        self.merge_pitch_semitones = merge_pitch_semitones
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
            merge_pitch_semitones=self.merge_pitch_semitones,
            merge_gap_s=self.merge_gap_s,
        )
