"""Gold names a pitch, and the decoder reads the same one back.

The two halves of the NLL have to agree about what a label *means*. Gold writes
``note@p``; the DP scores it against the pitch head's log-probability of ``p``; the
decoder turns the winning label back into a MIDI number. A disagreement anywhere in that
chain is silent -- every label is a valid integer, every shape matches, the loss falls --
and it trains the model toward a pitch nobody asked for.

The specific trap: ``BINARY``'s ``NOTE`` is 0, and under a pitch alphabet 0 is *the
lowest representable semitone*. So a gold built with the default alphabet where a wide
one is meant labels the entire corpus E1, and nothing raises.
"""

import numpy as np
import pytest

from stemify.music.notes import NoteEvent
from stemify.semicrf.labelset import BINARY, LabelSet

from spanscorer import CueScorer

from refs import gold

BINS = 144
LABELS = LabelSet.from_bins(BINS)          # MIDI 27..74
DT, T = 0.0348, 200
FT = np.arange(T) * DT


def _notes(pitches: list[int]) -> list[NoteEvent]:
    """One note per second, half a second long -- gaps wide enough to stay rests."""
    return [
        NoteEvent(onset=k * 1.0, offset=k * 1.0 + 0.5, midi_pitch=float(p))
        for k, p in enumerate(pitches)
    ]


def test_gold_labels_each_note_with_its_own_pitch():
    pitches = [40, 45, 52, 33]
    segs = gold(FT, _notes(pitches), dmax=115, labels=LABELS)
    named = [LABELS.midi_of(lab) for lab, _, _ in segs if LABELS.is_note(lab)]
    assert named == pitches


def test_gold_under_the_default_alphabet_is_still_binary():
    """The default has to keep meaning what it meant, or every existing checkpoint's
    gold changes underneath it."""
    segs = gold(FT, _notes([40, 45]), dmax=115)
    assert {lab for lab, _, _ in segs} <= {0, BINARY.REST}


def test_a_gold_built_under_the_wrong_alphabet_is_wrong_and_quiet():
    """Not a guard -- a demonstration of why the alphabet is threaded rather than
    defaulted. Both calls succeed and return the same *shape*; only the labels differ,
    and under the wide alphabet the default's answer means E1 every time."""
    notes = _notes([40, 45, 52])
    wide = gold(FT, notes, dmax=115, labels=LABELS)
    defaulted = gold(FT, notes, dmax=115)
    assert [(i, j) for _, i, j in wide] == [(i, j) for _, i, j in defaulted]
    assert {lab for lab, _, _ in defaulted if lab != BINARY.REST} == {0}
    assert LABELS.midi_of(0) == LABELS.lo != 40


def test_pitches_outside_the_alphabet_clamp_rather_than_raise():
    """A corpus note below the head's lowest bin has to land *somewhere*: the alphabet
    is derived from the pitch axis, so anything outside it is unrepresentable evidence,
    not a data error. Clamping keeps the tiling total, which the telescoping needs."""
    segs = gold(FT, _notes([LABELS.lo - 5]), dmax=115, labels=LABELS)
    assert [LABELS.midi_of(lab) for lab, _, _ in segs if LABELS.is_note(lab)] == [LABELS.lo]


def test_the_decoder_reads_the_pitch_off_the_label(monkeypatch):
    """Round trip: gold -> segments -> notes, with no contour anywhere in the chain.

    Under a pitch alphabet the label *is* the answer, so a decode needs no contour at
    all -- which is also what makes this test possible without synthesising audio."""
    from stemify.semicrf.decoder import SemiCRFDecoder
    from stemify.semicrf.evidence import Evidence
    from stemify.semicrf.neural import NeuralSpanScorer

    pitches = [40, 45, 52, 33]
    segs = gold(FT, _notes(pitches), dmax=115, labels=LABELS)
    scorer = NeuralSpanScorer(LABELS, trunk_dim=4, dmax_s=115 * DT, dmax_frames=115)
    dec = SemiCRFDecoder(scorer=scorer)
    import torch
    ev = Evidence(
        onset_logit=torch.zeros(1, T), voiced_c=torch.zeros(1, T + 1),
        rest_c=torch.zeros(1, T + 1), lengths=[T], frame_times=[FT], contours=None,
    )
    notes = dec._segs_to_notes(segs, ev)
    assert [n.midi_pitch for n in notes] == [float(p) for p in pitches]
    # Onsets land on the frame grid, so they are the reference times to within a hop.
    assert all(abs(n.onset - k * 1.0) <= DT for k, n in enumerate(notes))


def test_a_binary_alphabet_still_needs_a_contour():
    from stemify.semicrf.decoder import SemiCRFDecoder
    from stemify.semicrf.evidence import Evidence
    import torch

    ev = Evidence(
        onset_logit=torch.zeros(1, T), voiced_c=torch.zeros(1, T + 1),
        rest_c=torch.zeros(1, T + 1), lengths=[T], frame_times=[FT], contours=None,
    )
    with pytest.raises(ValueError, match="does not name them"):
        SemiCRFDecoder(CueScorer())._segs_to_notes([(0, 0, 10)], ev)
