"""``LabelSet``: the segment alphabet, and the two things it must never get wrong.

The alphabet is an integer encoding shared by the recurrence, the gold builder, the
scorer and the decoder. Both failure modes here are silent -- a label that maps to the
wrong pitch decodes a plausible note at the wrong place on the axis, and an alphabet
wider than the head's output indexes evidence that does not exist -- so they are worth
pinning rather than inferring from a DP test that would still pass.
"""

import pytest

from stemify.core.features import BINS_PER_SEMITONE, N_BINS, harmonic_shifts
from stemify.semicrf.constants import NOTE, REST, START
from stemify.semicrf.labelset import BINARY, LabelSet


def test_the_alphabet_is_one_label_per_pitch_then_rest():
    labels = LabelSet(lo=28, hi=31)          # E1..G1
    assert labels.n_pitches == 4
    assert [labels.label_of(m) for m in range(28, 32)] == [0, 1, 2, 3]
    assert labels.REST == 4                  # just past the pitch block
    assert labels.n_labels == 5
    assert labels.START == 5                 # one past the alphabet
    assert labels.n_rows == 6                # transitions are (n_rows, n_labels)


def test_pitch_labels_round_trip_and_rest_has_none():
    labels = LabelSet(lo=28, hi=74)
    for midi in range(28, 75):
        assert labels.midi_of(labels.label_of(midi)) == midi
    assert labels.label_of(40) != labels.REST
    with pytest.raises(ValueError, match="REST has no pitch"):
        labels.midi_of(labels.REST)


def test_out_of_range_pitches_clamp_rather_than_raise():
    # Gold is built from reference notes, and one note off the head's axis should cost
    # that note rather than the run. Clamping is bounded and shows up as a systematic
    # error at the extremes; raising would fail mid-epoch on a corpus quirk.
    labels = LabelSet(lo=28, hi=74)
    assert labels.midi_of(labels.label_of(200)) == 74
    assert labels.midi_of(labels.label_of(-5)) == 28


def test_the_range_is_derived_from_the_head_and_stays_inside_its_axis():
    # The load-bearing property: every pitch label must index a bin the head actually
    # emits. A separately-declared range is free to drift past a cropped axis, and the
    # label that does simply never wins -- no error, just a pitch the model cannot name.
    for n_bins in (N_BINS, N_BINS - max(harmonic_shifts((1, 2, 3, 4)))):
        labels = LabelSet.from_bins(n_bins)
        assert labels.n_pitches == n_bins // BINS_PER_SEMITONE
        assert 0 <= labels.bin_of(0)
        assert labels.bin_of(labels.n_pitches - 1) < n_bins


def test_harmonic_stacking_narrows_the_alphabet():
    # `_ConvFrontend.stack` crops the fundamental axis by the topmost harmonic's shift,
    # so a model trained with --harmonics names fewer pitches than one without. Deriving
    # from the head is what keeps the two in step.
    plain = LabelSet.from_bins(N_BINS)
    stacked = LabelSet.from_bins(N_BINS - max(harmonic_shifts((1, 2, 3, 4))))
    assert stacked.n_pitches == 48 and plain.n_pitches == 72
    assert stacked.lo == plain.lo          # both start at FMIN
    assert stacked.hi < plain.hi           # the crop takes the top


def test_binary_reproduces_the_original_integers_not_just_the_width():
    # The property that protects every existing semi-CRF checkpoint. `LinearSpanScorer.
    # trans` is indexed by these integers and rides in the state dict, so an alphabet
    # that merely had the right *width* but ordered rest first would load every saved
    # model with its transition rows permuted -- no error, no shape mismatch, a
    # different model. Pitches must come first.
    assert BINARY.label_of(0) == NOTE == 0
    assert BINARY.REST == REST == 1
    assert BINARY.START == START == 2
    assert BINARY.n_labels == 2
    assert BINARY.n_rows == 3


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="empty pitch range"):
        LabelSet(lo=40, hi=39)


def test_a_sub_semitone_axis_has_no_alphabet_to_derive():
    with pytest.raises(ValueError, match="under one semitone"):
        LabelSet.from_bins(BINS_PER_SEMITONE - 1)
