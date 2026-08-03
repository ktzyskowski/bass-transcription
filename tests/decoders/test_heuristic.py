"""GreedyDecoder / group_contour -- the greedy decode core, no audio.

group_contour is exercised with synthetic contours and hand-picked cues; the decoder
is exercised through Cues (a per-frame onset curve + optional voicing curve), so the
peak-pick + threshold + group path is covered without a model.
"""

import numpy as np

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.greedy import GreedyDecoder, group_contour

T = 100
TIMES = np.arange(T) * 0.01  # 10 ms grid, 1 s


def _held(hz: float = 100.0) -> Contour:
    return Contour(times=TIMES, freqs=np.full(T, hz))


def test_group_contour_splits_a_held_note_at_an_onset():
    # One held pitch, an onset half-way -> a same-pitch re-articulation = two notes.
    notes = group_contour(_held(), onsets=np.array([0.5]), merge_gap_s=None)
    assert len(notes) == 2


def test_group_contour_empty_contour_is_no_notes():
    assert group_contour(Contour.empty()) == []


def test_decoder_peak_picks_the_onset_curve():
    onset = np.zeros(T, dtype=np.float32)
    onset[50] = 1.0  # a single onset spike -> one same-pitch split
    cues = Cues(contour=_held(), onset=onset, frame_times=TIMES, voicing=None)
    notes = GreedyDecoder(merge_gap_s=None).decode(cues)
    assert len(notes) == 2
    assert abs(notes[1].onset - 0.5) < 0.02  # the split lands at the spike


def test_decoder_thresholds_the_voicing_curve_into_rests():
    # Contour voiced throughout, but the voicing curve dips unvoiced mid-track ->
    # the decoder places a rest there and emits two notes.
    voicing = np.ones(T, dtype=np.float32)
    voicing[40:60] = 0.0
    cues = Cues(
        contour=_held(), onset=np.zeros(T, dtype=np.float32),
        frame_times=TIMES, voicing=voicing,
    )
    assert len(GreedyDecoder(merge_gap_s=None).decode(cues)) == 2


def test_decoder_without_voicing_falls_back_to_contour():
    # No voicing curve: a contour with an unvoiced gap (freqs == 0) still splits.
    freqs = np.full(T, 100.0)
    freqs[40:60] = 0.0
    cues = Cues(
        contour=Contour(times=TIMES, freqs=freqs),
        onset=np.zeros(T, dtype=np.float32), frame_times=TIMES, voicing=None,
    )
    assert len(GreedyDecoder(merge_gap_s=None).decode(cues)) == 2
