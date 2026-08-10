"""``GreedyDecoder``: the greedy decode (no learning).

Turns per-frame ``Frames`` into notes by splitting the pitch contour at a rest (an
unvoiced run) and at an onset (a re-articulation the contour alone cannot show),
then rejoining fragments a spurious cue split. The rest/onset cues over-fire (flux
triggers on mid-sustain noise; pyin drops out for a frame), so the optional
``merge_gap_s`` pass undoes the dominant source of over-segmentation, trading
against recall.

A *pitch change* used to be a third splitting cue, and is now off by default -- see
``PITCH_SPLIT_SEMITONES`` for why, and for the one encoder that still needs it.

The pure core is ``group_contour`` -- no audio, no thresholds, taking onset *times*
and a voiced mask directly -- so a boundary case is testable with a synthetic
contour and hand-picked cues. ``GreedyDecoder.decode`` is the ``Decoder`` wrapper:
it peak-picks the onset curve and thresholds the voicing curve out of the ``Frames``,
then groups. That thresholding is exactly what a ``SemiCRFDecoder`` does *not* do --
it keeps the soft probabilities -- which is why the two are siblings.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks

from stemify.music.contour import Contour
from stemify.core.frames import Frames
from stemify.core.decoders.base import Decoder
from stemify.music.grids import resample_mask
from stemify.music.notes import NoteEvent

# Below this, a "note" is a tracker blip (onset transient, a frame of octave wobble)
# rather than a played note. Enforced against the *contour* grid, which is what makes
# the right value depend on where the contour came from: a threshold set against pyin's
# 23.2 ms frames admits far more short fragments on the CQT's 11.6 ms grid, so this has
# to be re-swept whenever the pitch source changes.
#
# Sits at the *middle* of a flat plateau rather than at its argmax -- validation and
# test disagreed about which end was better, which is what riding a boundary looks like
# on a split this size. Note this discards real notes shorter than it, a permanent
# recall cap accepted because fragments that short are more often false positives than
# notes; revisit if short-note recall ever becomes the binding constraint.
MIN_NOTE_S = 0.10

# A sustained jump this far from the note's running pitch starts a new note.
# ``None`` -- the default -- means a pitch change never places a note on its own.
#
# This was the only cue a DSP pipeline had for a new note, and it became an active
# liability the moment a trained onset head existed. pyin reaches a note's true pitch
# well *after* the attack -- often past the match tolerance -- so a real onset makes the
# contour walk a staircase through intermediate pitches, and every step of that
# staircase split. Worse, ``group_contour`` had already placed the note correctly at the
# onset: the split then cut a sub-``MIN_NOTE_S`` fragment off its front, ``emit``
# dropped that, and the note that survived started late enough to miss tolerance. One
# reference note, one false negative and one interior-offpitch false positive.
#
# The sweep is monotone from 0.8 out to "off" with no interior optimum, which is the
# shape of a superseded rule rather than a mistuned one. See docs/experiments.md.
PITCH_SPLIT_SEMITONES: float | None = None

# What an encoder with no trained onset head falls back to. ``ClassicalEncoder`` has only a
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
# Both thresholds are genuine operating points rather than leftovers, though
# MIN_ONSET_GAP_S is *inert* for a trained onset head -- its peaks are already well
# separated -- and is kept for the DSP path, whose flux spikes are exactly what it
# guards against. VOICING_THRESHOLD trades the way MIN_NOTE_S does: a lower bar marks
# more frames as sounding, so the mask punches fewer spurious rests through a held note
# and fragmentation false positives fall, at some cost to how crisply rests are placed.
#
# This is *not* the threshold that gates the learned contour -- that is
# ``models.heads.CONTOUR_VOICING_THRESHOLD``, deliberately separate. The two were one
# constant once, which meant retuning here silently retuned the contour as well; they
# were swept together and their values only hold as a pair. See docs/experiments.md.
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
    times: np.ndarray = frame_times[peaks]
    return times


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


class GreedyDecoder(Decoder):
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

    def decode(self, frames: Frames) -> list[NoteEvent]:
        """Peak-pick the onset curve and threshold the voicing curve out of ``frames``,
        align voicing to the contour grid, then group. With no voicing head, grouping
        falls back to the contour's own ``freqs > 0``."""
        if frames.contour is None:
            raise ValueError(
                "GreedyDecoder names every note's pitch from the contour, and these "
                "Frames carry none. Encode with a pitch source (pyin, or a model with "
                "a FrequencyHead), or decode with SemiCRFDecoder, which reads the "
                "pitch head's own per-bin logits instead of a point estimate."
            )
        # `Frames` holds numpy at decode and tensors under training, and this decoder is
        # the numpy half: scipy peak-picking has no tensor path. `asarray` is a view on
        # a CPU tensor and *raises* on a CUDA one, which is the right answer -- evidence
        # still on the accelerator has taken a wrong turn, not a slow one.
        onset = np.asarray(frames.onset)
        voicing = None if frames.voicing is None else np.asarray(frames.voicing)
        onsets = onset_times_from_probs(
            onset, frames.frame_times, self.onset_threshold, self.min_onset_gap_s
        )
        voiced = None
        if voicing is not None:
            mask = voicing > self.voicing_threshold
            voiced = resample_mask(
                frames.frame_times, mask, frames.contour.times
            ).astype(bool)
        return group_contour(
            frames.contour,
            onsets,
            voiced,
            min_note_s=self.min_note_s,
            pitch_split_semitones=self.pitch_split_semitones,
            merge_pitch_semitones=self.merge_pitch_semitones,
            merge_gap_s=self.merge_gap_s,
        )
