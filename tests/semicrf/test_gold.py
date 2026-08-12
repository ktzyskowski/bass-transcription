"""``gold_segments``: the tiling it must produce, and where a rest comes from.

Half the NLL is ``score(gold)``, so a wrong gold is not a wrong *loss* -- it is a
different objective, minimised correctly. Nothing downstream can see it: the tiling
still tiles, every index is still in range, the loss still falls.

Two properties carry this file.

**The tiling invariant**, asserted over randomised layouts: the segments partition
``[0, T)`` exactly -- contiguous, no gap, no overlap, every length in ``[1, dmax]``.
Checkable without reimplementing the labelling rule, which is what makes it an oracle
rather than a second opinion. It is also what the whole DP rests on: the recurrence sums
over tilings, so a gold that is not one is scored against spans ``log Z`` never
enumerated, and the NLL then has no lower bound.

**The agreement invariant**: *no frame gold calls ``rest`` is a frame the reference
calls voiced.* This is the one that was violated. Gold used to answer "when does a note
stop sounding" from a flat 120 ms constant while the voicing target measured the decay,
and on 40 real tracks 28% of gold's rest frames landed on frames the reference called
voiced. Since the voicing head's output *is* the DP's rest evidence, gold was scored
against a signal it contradicted -- worth a median 8.8 nats per rest span, in favour of
merging the note through it. A rest is now *defined* as inaudible, so the two cannot
disagree; this test is what keeps it that way.

The fixtures build ``Reference`` by hand rather than through ``tests/refs.py``, because
the mask is the subject here and a helper would be supplying the answer.
"""

import numpy as np
import pytest

from stemify.music.notes import NoteEvent
from stemify.music.reference import Reference
from stemify.semicrf.constants import NOTE, REST
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY, LabelSet

DT = 0.032


def _times(n: int) -> np.ndarray:
    return np.arange(n) * DT


def _note(onset: float, offset: float, midi: float = 45.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=offset, midi_pitch=midi)


def _ref(n: int, notes: list[NoteEvent], ring_s: float = 0.0) -> Reference:
    """Audible across each note's MIDI span plus ``ring_s``, capped at the next onset."""
    times = _times(n)
    voiced = np.zeros(n, dtype=bool)
    ordered = sorted(notes, key=lambda x: x.onset)
    for k, note in enumerate(ordered):
        nxt = ordered[k + 1].onset if k + 1 < len(ordered) else np.inf
        voiced |= (times >= note.onset) & (times < min(note.offset + ring_s, nxt))
    return Reference(frame_times=times, voiced=voiced, notes=tuple(ordered))


def _assert_tiles(segs, n_frames: int, dmax: int) -> None:
    """``segs`` partitions ``[0, n_frames)`` into spans of 1..dmax."""
    if n_frames == 0:
        assert segs == [], f"an empty grid has nothing to tile, got {segs}"
        return
    assert segs, "a non-empty grid must be tiled"
    cursor = 0
    for k, (_, start, end) in enumerate(segs):
        assert start == cursor, (
            f"segment {k} starts at {start}, but the previous ended at {cursor} -- "
            "gold is not contiguous"
        )
        assert 1 <= end - start <= dmax, (
            f"segment {k} is {end - start} frames, outside [1, {dmax}]; the scan "
            "enumerates no such span, so log Z cannot contain this gold"
        )
        cursor = end
    assert cursor == n_frames, f"gold covers [0, {cursor}), not [0, {n_frames})"


def _frames_labelled(segs, n_frames: int, want) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for label, start, end in segs:
        if want(label):
            mask[start:end] = True
    return mask


# --------------------------------------------------------------------------- #
# The agreement invariant -- the one that was broken                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ring_s", [0.0, 0.05, 0.20, 0.60])
def test_no_rest_frame_is_one_the_reference_calls_voiced(seed, ring_s):
    """**The invariant this module exists to hold.**

    A rest is where the bass is inaudible. The DP scores a rest span with
    ``Σ log(1 - P(voiced))`` read off the very head the voicing target trains, so a rest
    placed on a ringing frame is a claim gold's own evidence contradicts -- and the
    cheaper alternative is always to merge the note through it.

    Swept across ring lengths because that is the axis the old flat 120 ms constant got
    wrong: it was right only for notes whose decay happened to land near 120 ms.
    """
    rng = np.random.default_rng(seed)
    n_frames, dmax = 300, 40
    span = float(_times(n_frames)[-1])
    notes, t = [], 0.1
    while t < span - 0.5:
        dur = float(rng.uniform(0.10, 0.5))
        notes.append(_note(t, t + dur, 40.0 + int(rng.integers(0, 8))))
        t += dur + float(rng.uniform(0.02, 0.5))

    reference = _ref(n_frames, notes, ring_s)
    segs = gold_segments(reference, dmax)
    _assert_tiles(segs, n_frames, dmax)

    rest = _frames_labelled(segs, n_frames, lambda label: label == REST)
    offenders = int((rest & reference.voiced).sum())
    assert offenders == 0, (
        f"{offenders} rest frames sit where the reference says the bass is still "
        f"sounding (ring {ring_s * 1000:.0f} ms) -- the DP would be paying to merge "
        "the note through every one of them"
    )


def test_a_note_owns_its_ring_out():
    """The positive half: frames after note-off where the bass is still audible belong
    to the note, not to a rest. That is the whole content of the change."""
    notes = [_note(0.10, 0.40), _note(1.20, 1.50)]
    reference = _ref(120, notes, ring_s=0.30)
    segs = gold_segments(reference, 40)
    _assert_tiles(segs, 120, 40)
    note_frames = _frames_labelled(segs, 120, lambda label: label != REST)
    at = int(np.searchsorted(_times(120), 0.55))   # past note-off, inside the ring
    assert reference.voiced[at] and note_frames[at], (
        "a frame inside the first note's ring-out was labelled rest"
    )


def test_a_gap_the_bass_does_not_survive_is_a_rest():
    """The other half. A note that stops dead leaves real silence and gold must say so,
    or every gap becomes note and ``rest`` stops meaning anything."""
    notes = [_note(0.10, 0.40), _note(1.20, 1.50)]
    segs = gold_segments(_ref(120, notes, ring_s=0.0), 40)
    _assert_tiles(segs, 120, 40)
    assert any(label == REST for label, _, _ in segs)
    rest = _frames_labelled(segs, 120, lambda label: label == REST)
    assert rest[int(np.searchsorted(_times(120), 0.80))]   # mid-gap, silent


def test_a_ring_reaching_the_next_onset_gives_a_contiguous_note_transition():
    """Legato: the bass never stops, so there is no rest between the notes and the DP
    sees note->note. Training on a phantom rest here is what taught the CRF to avoid
    note->note and collapse legato runs."""
    notes = [_note(0.10, 0.40, 45.0), _note(0.60, 0.90, 47.0)]
    segs = gold_segments(_ref(80, notes, ring_s=1.0), 40, LabelSet(lo=40, hi=52))
    _assert_tiles(segs, 80, 40)
    interior = [s for s in segs if s[1] > 0 and s[2] < 80]
    assert all(label != REST for label, _, _ in interior), (
        f"a rest appeared between two notes the bass sustains through: {segs}"
    )


# --------------------------------------------------------------------------- #
# The tiling invariant                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
def test_gold_tiles_the_grid_for_arbitrary_note_layouts(seed):
    """Random onsets, durations and gaps -- including overlaps, zero-length notes and
    notes running past the end of the grid, all of which reach this from real MIDI."""
    rng = np.random.default_rng(seed)
    n_frames, dmax = 200, 12
    span = float(_times(n_frames)[-1])
    notes = []
    for _ in range(int(rng.integers(0, 12))):
        onset = float(rng.uniform(-0.1, span + 0.1))
        notes.append(_note(onset, onset + float(rng.exponential(0.4))))
    rng.shuffle(notes)                      # unsorted input is the caller's normal case
    _assert_tiles(gold_segments(_ref(n_frames, notes, 0.1), dmax), n_frames, dmax)


@pytest.mark.parametrize("n_notes", [0, 1, 2, 5])
@pytest.mark.parametrize("dmax", [1, 2, 7, 40])
def test_gold_tiles_at_every_segment_cap(n_notes, dmax):
    """``dmax`` bounds the scan's candidate starts, so gold must respect it whatever it
    is -- including ``dmax=1``, where every segment is one frame."""
    n_frames = 60
    notes = [_note(0.2 + 0.5 * k, 0.6 + 0.5 * k) for k in range(n_notes)]
    _assert_tiles(gold_segments(_ref(n_frames, notes), dmax), n_frames, dmax)


def test_an_empty_grid_yields_no_segments():
    """``[0, 0)`` has one tiling and it is the empty one. Reached by a zero-length chunk
    at a track's end; an off-by-one emits a negative-length span that indexes backwards
    into the evidence."""
    _assert_tiles(gold_segments(_ref(0, [_note(0.0, 1.0)]), 8), 0, 8)


def test_notes_outside_the_grid_do_not_escape_it():
    notes = [_note(-5.0, -1.0), _note(0.1, 0.3), _note(0.5, 99.0)]
    segs = gold_segments(_ref(20, notes, 0.05), 8)
    _assert_tiles(segs, 20, 8)
    assert all(0 <= start < end <= 20 for _, start, end in segs)


def test_a_segment_cap_below_one_frame_is_refused():
    """``dmax: int`` admits 0 and negatives, and the filler advances by ``dmax``. At 0
    it never advances -- the old version looped forever, growing the segment list until
    it exhausted memory."""
    for dmax in (0, -1, -7):
        with pytest.raises(ValueError, match="at least 1 frame"):
            gold_segments(_ref(20, [_note(0.1, 0.3)]), dmax)


# --------------------------------------------------------------------------- #
# Labels and placement                                                         #
# --------------------------------------------------------------------------- #
def test_a_pitch_alphabet_names_each_notes_own_semitone():
    labels = LabelSet(lo=40, hi=52)
    pitches = [43.0, 47.0, 51.0]
    notes = [_note(0.2 + 0.6 * k, 0.5 + 0.6 * k, p) for k, p in enumerate(pitches)]
    segs = gold_segments(_ref(400, notes), 40, labels)
    _assert_tiles(segs, 400, 40)
    named = [labels.midi_of(label) for label, _, _ in segs if labels.is_note(label)]
    assert named == pytest.approx(pitches)


def test_the_default_alphabet_is_binary_and_labels_every_note_zero():
    """Under a pitch alphabet 0 is *the lowest representable semitone*, so defaulting
    the alphabet where a wide one is meant does not raise, does not change a shape, and
    trains every note toward E1."""
    segs = gold_segments(_ref(100, [_note(0.2, 0.9, 60.0)]), 20)
    assert {label for label, _, _ in segs} <= {NOTE, BINARY.REST}
    assert any(label == NOTE for label, _, _ in segs)
    assert BINARY.label_of(60.0) == NOTE == 0


def test_an_onset_lands_on_the_nearest_frame_not_the_next_one():
    """The same rule ``music.labels.onset_targets`` uses.

    Rounding up is the right answer to "which frames does this note occupy" and the
    wrong one for "when does it start" -- the decoder reads a note's time back off this
    index, so it made every emitted onset late by up to a frame, one-sided.
    """
    n = 60
    times = _times(n)
    onset = float(times[10] + 0.35 * DT)     # nearest is 10; the next frame up is 11
    segs = gold_segments(_ref(n, [_note(onset, onset + 0.4)]), 40)
    first_note = next(start for label, start, _ in segs if label == NOTE)
    assert first_note == 10, f"onset snapped to frame {first_note}, expected 10"


def test_a_note_longer_than_dmax_becomes_more_note_not_a_rest():
    """The DP cannot hold it as one span. Cutting it short and resting the tail would
    teach silence inside a sounding note; consecutive same-label spans are what a unison
    transition already models."""
    dmax = 10
    segs = gold_segments(_ref(200, [_note(0.0, float(_times(200)[-1]) + 1.0)]), dmax)
    _assert_tiles(segs, 200, dmax)
    assert all(label == NOTE for label, _, _ in segs), (
        f"a held note past dmax produced non-note segments: "
        f"{[s for s in segs if s[0] != NOTE][:3]}"
    )
