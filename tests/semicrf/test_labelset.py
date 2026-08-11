"""``LabelSet``: the segment alphabet, and the two things it must never get wrong.

The alphabet is an integer encoding shared by the recurrence, the gold builder, the
scorer and the decoder. Both failure modes here are silent -- a label that maps to the
wrong pitch decodes a plausible note at the wrong place on the axis, and an alphabet
wider than the head's output indexes evidence that does not exist -- so they are worth
pinning rather than inferring from a DP test that would still pass.
"""

import pytest

from stemify.audio.features import BINS_PER_SEMITONE, N_BINS, harmonic_shifts
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
    # The property that protects every existing semi-CRF checkpoint. A scorer's
    # transition table is indexed by these integers and rides in the state dict, so
    # an alphabet with the right *width* but rest ordered first would load every saved
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


def test_pitch_logprob_cumsum_scores_the_right_semitone():
    """The one pitch-*valued* quantity the DP has, and the reason to widen the alphabet.

    Every other span feature measures change or disagreement and is invariant to which
    pitch is named -- so without this, `note@p` and `note@q` score identically and log Z
    spreads its mass over every relabelling of the same segmentation."""
    import torch

    from stemify.semicrf.cumsums import pitch_logprob_cumsum

    labels = LabelSet.from_bins(144)
    want = labels.label_of(40)                       # E2
    logits = torch.full((20, 144), -5.0)
    centre = labels.bin_of(want)
    logits[:, centre - 1 : centre + 2] = 20.0

    c = pitch_logprob_cumsum(logits, labels.n_pitches)
    assert c.shape == (21, labels.n_pitches)
    assert bool((c[0] == 0).all()), "cumsum needs a zero row so C[j]-C[i] spans [i, j)"

    mean_lp = (c[20] - c[0]) / 20                    # the span's mean log-probability
    assert int(mean_lp.argmax()) == want
    assert float(mean_lp.max()) > -0.05              # ~certain at the right semitone
    assert float(mean_lp[0]) < -5.0                  # and far from it elsewhere


def test_pitch_logprob_pools_semitones_rather_than_picking_a_centre_bin():
    """`two_hot` splits a note's mass across the two bins straddling its true
    fractional pitch, so reading the centre bin alone under-counts every note that is
    not exactly at a semitone -- systematically, and worst for the in-tune ones.
    Pooling by logsumexp is the marginal 'somewhere in this semitone'."""
    import torch

    from stemify.semicrf.cumsums import pitch_logprob_cumsum

    labels = LabelSet.from_bins(144)
    want = labels.label_of(40)
    centre = labels.bin_of(want)
    # Mass split evenly across two neighbouring bins of the same semitone, nothing on
    # the centre: a centre-bin reading would call this pitch unlikely.
    logits = torch.full((4, 144), -20.0)
    logits[:, centre - 1] = 5.0
    logits[:, centre + 1] = 5.0

    mean_lp = (pitch_logprob_cumsum(logits, labels.n_pitches)[4]) / 4
    assert int(mean_lp.argmax()) == want
    assert float(mean_lp.max()) > -0.2


def test_a_head_narrower_than_the_alphabet_is_refused():
    import pytest as _pytest
    import torch

    from stemify.semicrf.cumsums import pitch_logprob_cumsum

    # The alphabet is derived from the head, so a mismatch means they have drifted --
    # and the silent version scores pitches against evidence that does not exist.
    with _pytest.raises(ValueError, match="have drifted"):
        pitch_logprob_cumsum(torch.zeros(3, 30), n_pitches=48)
