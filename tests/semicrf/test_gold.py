"""``gold_segments``: the tiling it must produce, and the audible-gap rule.

Half the NLL is ``score(gold)``, so a wrong gold is not a wrong *loss* -- it is a
different objective, minimised correctly. Nothing downstream can see it: the tiling
still tiles, every index is still in range, the loss still falls.

The module is pure numpy and had two tests, both single hand-built cases. What is added
here is the **tiling invariant**, asserted over randomised note layouts:

    the segments partition [0, T) exactly -- contiguous, no gap, no overlap,
    every length in [1, dmax]

That is checkable without reimplementing the labelling rule, which is what makes it an
oracle rather than a second opinion. It is also the property the whole DP rests on: the
recurrence sums over tilings, so a gold that is not one is being scored against spans
``log Z`` never enumerated -- and the NLL then has no lower bound.

The gap rule itself is pinned separately, on both sides of its threshold, because a
threshold is where a comparison operator flips silently.
"""

import numpy as np
import pytest

from stemify.semicrf.constants import GAP_REST_S, NOTE, REST
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.music.notes import NoteEvent

DT = 0.032


def _times(n: int) -> np.ndarray:
    return np.arange(n) * DT


def _note(onset: float, offset: float, midi: float = 45.0) -> NoteEvent:
    return NoteEvent(onset=onset, offset=offset, midi_pitch=midi)


def _assert_tiles(segs, n_frames: int, dmax: int) -> None:
    """The invariant: ``segs`` partitions ``[0, n_frames)`` into spans of 1..dmax.

    Written as one helper because every case below needs it and because stating it once
    is the point -- this is the contract, not a detail of any particular layout.
    """
    if n_frames == 0:
        assert segs == [], f"an empty grid has nothing to tile, got {segs}"
        return
    assert segs, "a non-empty grid must be tiled"
    cursor = 0
    for k, (_, start, end) in enumerate(segs):
        assert start == cursor, (
            f"segment {k} starts at {start}, but the previous one ended at {cursor} -- "
            "gold is not contiguous"
        )
        assert 1 <= end - start <= dmax, (
            f"segment {k} is {end - start} frames, outside [1, {dmax}]; the scan "
            "enumerates no such span, so log Z cannot contain this gold"
        )
        cursor = end
    assert cursor == n_frames, f"gold covers [0, {cursor}), not [0, {n_frames})"


# --------------------------------------------------------------------------- #
# The tiling invariant, over layouts nobody chose                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
def test_gold_tiles_the_grid_for_arbitrary_note_layouts(seed):
    """Random onsets, durations and gaps -- including overlaps, zero-length notes and
    notes running past the end of the grid, all of which reach this from real MIDI."""
    rng = np.random.default_rng(seed)
    n_frames, dmax = 200, 12
    times = _times(n_frames)
    span = float(times[-1])

    notes = []
    for _ in range(int(rng.integers(0, 12))):
        onset = float(rng.uniform(-0.1, span + 0.1))
        notes.append(_note(onset, onset + float(rng.exponential(0.4))))
    rng.shuffle(notes)                      # unsorted input is the caller's normal case

    _assert_tiles(gold_segments(times, notes, dmax), n_frames, dmax)


@pytest.mark.parametrize("n_notes", [0, 1, 2, 5])
@pytest.mark.parametrize("dmax", [1, 2, 7, 40])
def test_gold_tiles_at_every_segment_cap(n_notes, dmax):
    """``dmax`` bounds the scan's candidate starts, so gold has to respect it whatever
    it is set to -- including ``dmax=1``, where every segment is one frame and the rest
    filler does all the work."""
    n_frames = 60
    times = _times(n_frames)
    notes = [_note(0.2 + 0.5 * k, 0.6 + 0.5 * k) for k in range(n_notes)]
    _assert_tiles(gold_segments(times, notes, dmax), n_frames, dmax)


def test_an_empty_grid_yields_no_segments():
    """``[0, 0)`` has one tiling and it is the empty one. Reached by a zero-length
    chunk at a track's end, and an off-by-one here would emit a span of negative
    length that indexes backwards into the evidence."""
    _assert_tiles(gold_segments(_times(0), [_note(0.0, 1.0)], 8), 0, 8)


def test_notes_outside_the_grid_do_not_escape_it():
    """A note whose offset runs past the last frame, and one whose onset precedes the
    first. Gold is a tiling of the frames that *exist*."""
    times = _times(20)
    notes = [_note(-5.0, -1.0), _note(0.1, 0.3), _note(0.5, 99.0)]
    segs = gold_segments(times, notes, 8)
    _assert_tiles(segs, 20, 8)
    assert all(0 <= start < end <= 20 for _, start, end in segs)


# --------------------------------------------------------------------------- #
# The audible-gap rule, on both sides of its threshold                         #
# --------------------------------------------------------------------------- #
def test_a_gap_longer_than_the_threshold_becomes_a_rest():
    times = _times(200)
    gap = GAP_REST_S * 2
    segs = gold_segments(times, [_note(0.2, 0.9), _note(0.9 + gap, 1.8)], 40)
    _assert_tiles(segs, 200, 40)
    labels = [label for label, _, _ in segs]
    assert labels.count(NOTE) >= 2 and REST in labels


def test_a_gap_shorter_than_the_threshold_is_closed_into_the_previous_note():
    """A MIDI note-off artifact: the audio rings on, so there is no audible silence.

    Training on the raw gap teaches a rest the audio never contains *and* a spurious
    ``note->note`` penalty -- the under-segmentation trap that collapses legato runs.
    """
    times = _times(200)
    gap = GAP_REST_S / 3
    segs = gold_segments(times, [_note(0.2, 0.9), _note(0.9 + gap, 1.8)], 40)
    _assert_tiles(segs, 200, 40)
    note_at = [k for k, (label, _, _) in enumerate(segs) if label == NOTE]
    between = segs[note_at[0] : note_at[-1] + 1]
    assert all(label == NOTE for label, _, _ in between), (
        f"a {gap * 1000:.0f} ms gap (threshold {GAP_REST_S * 1000:.0f} ms) became a "
        f"rest: {between}"
    )


@pytest.mark.parametrize("factor,closes", [(0.5, True), (0.95, True), (1.5, False)])
def test_the_threshold_is_where_the_rule_flips(factor, closes):
    """Swept across ``GAP_REST_S`` rather than asserted at one point, because the
    boundary is where the frame grid's own quantisation can move the answer by a frame.

    Exact equality is deliberately not pinned. ``gap_s`` is a difference of two float
    frame times, so ``gap_s == GAP_REST_S`` is unreachable in practice, and the
    docstring's rule ("shorter is an artifact, longer stays a rest") does not define
    the equal case -- a test there would be fixing a free choice at a measure-zero
    point. Swapping ``>`` for ``>=`` in the implementation is, for every input this can
    construct, the same function."""
    times = _times(400)
    gap = GAP_REST_S * factor
    segs = gold_segments(times, [_note(0.2, 0.9), _note(0.9 + gap, 1.8)], 40)
    _assert_tiles(segs, 400, 40)
    interior = segs[
        next(k for k, (lab, _, _) in enumerate(segs) if lab == NOTE)
        : len(segs) - next(k for k, (lab, _, _) in enumerate(reversed(segs))
                           if lab == NOTE)
    ]
    has_rest = any(label == REST for label, _, _ in interior)
    assert has_rest is not closes, (
        f"gap {gap * 1000:.0f} ms vs threshold {GAP_REST_S * 1000:.0f} ms: "
        f"expected {'no interior rest' if closes else 'an interior rest'}"
    )


def test_closing_a_gap_holds_the_previous_notes_label_not_the_next_ones():
    """The extension continues the note that was *already sounding*.

    Under a pitch alphabet this is the difference between extending an A and inventing
    one: both tile, both score, and only one is the audio. Invisible under ``BINARY``,
    where every note carries label 0 -- which is why it is checked at a wide alphabet.
    """
    labels = LabelSet(lo=40, hi=52)
    times = _times(400)
    gap = GAP_REST_S / 3
    first, second = 45.0, 50.0
    segs = gold_segments(
        times,
        [_note(0.2, 0.9, first), _note(0.9 + gap, 1.8, second)],
        40, labels,
    )
    _assert_tiles(segs, 400, 40)
    # The segment covering the gap is the one containing the frame just after 0.9.
    at = int(np.searchsorted(times, 0.9 + gap / 2))
    covering = next(label for label, start, end in segs if start <= at < end)
    assert covering == labels.label_of(first), (
        f"the gap was filled with {labels.midi_of(covering)}, not the note that was "
        f"sounding ({first})"
    )


# --------------------------------------------------------------------------- #
# Labels                                                                       #
# --------------------------------------------------------------------------- #
def test_a_pitch_alphabet_names_each_notes_own_semitone():
    labels = LabelSet(lo=40, hi=52)
    times = _times(400)
    pitches = [43.0, 47.0, 51.0]
    segs = gold_segments(
        times,
        [_note(0.2 + 0.6 * k, 0.5 + 0.6 * k, p) for k, p in enumerate(pitches)],
        40, labels,
    )
    _assert_tiles(segs, 400, 40)
    named = [labels.midi_of(label) for label, _, _ in segs if labels.is_note(label)]
    assert named == pytest.approx(pitches)


def test_the_default_alphabet_is_binary_and_labels_every_note_zero():
    """``BINARY``'s ``NOTE`` is 0, and under a pitch alphabet 0 is *the lowest
    representable semitone* -- so defaulting the alphabet where a wide one is meant
    does not raise, does not change a shape, and trains every note toward E1."""
    times = _times(100)
    segs = gold_segments(times, [_note(0.2, 0.9, 60.0)], 20)
    assert {label for label, _, _ in segs} <= {NOTE, BINARY.REST}
    assert any(label == NOTE for label, _, _ in segs)
    assert BINARY.label_of(60.0) == NOTE == 0


# --------------------------------------------------------------------------- #
# The segment cap                                                              #
# --------------------------------------------------------------------------- #
def test_a_note_longer_than_dmax_is_clipped_and_the_remainder_still_tiles():
    """A held note past the segment cap cannot be one span, and the DP has no way to
    represent it as two ``note`` spans without a ``note->note`` transition it would
    then be trained to expect. Gold clips it to ``dmax`` and fills the remainder --
    so the *tiling* survives, which is what the NLL needs.

    Pinned because the fill is a **rest**, i.e. a held note past ``dmax`` teaches
    silence in its own tail. ``DMAX_S`` is 4 s against a bass note that rarely reaches
    it, so this is a bound on how wrong that can get, not a live defect -- but it is a
    real asymmetry and it should fail loudly if ``DMAX_S`` is ever lowered.
    """
    times = _times(200)
    dmax = 10
    segs = gold_segments(times, [_note(0.0, float(times[-1]))], dmax)
    _assert_tiles(segs, 200, dmax)
    notes = [(start, end) for label, start, end in segs if label == NOTE]
    assert len(notes) == 1 and notes[0] == (0, dmax), (
        f"expected one dmax-long note span, got {notes}"
    )
    assert all(label == REST for label, start, _ in segs if start >= dmax)
