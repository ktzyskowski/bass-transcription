"""reference_aligned_cues: turn an item's reference + the encoder's Cues into aligned pairs.

This is where the grid alignment lives (onset impulses, voiced mask, contour onto
the reference grid), so it is exercised with a hand-built item and Cues -- no audio.
"""

import numpy as np

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.evaluation.cues import reference_aligned_cues
from stemify.notes import NoteEvent

FRAME_TIMES = np.arange(12) * 0.0116


class _FakeItem:
    def __init__(self, note_onset: float):
        self._onset = note_onset

    @property
    def name(self) -> str:
        return "fake"

    def reference(self) -> Contour:
        return Contour(times=FRAME_TIMES, freqs=np.full(FRAME_TIMES.size, 110.0))

    def reference_notes(self):
        return [NoteEvent(onset=self._onset, offset=self._onset + 0.05, midi_pitch=45.0)]


def _cues(with_voicing: bool) -> Cues:
    return Cues(
        contour=Contour(times=FRAME_TIMES, freqs=np.full(12, 110.0)),
        onset=np.zeros(12, dtype=np.float32),
        frame_times=FRAME_TIMES,
        voicing=np.ones(12, dtype=np.float32) if with_voicing else None,
    )


def test_onset_target_impulse_lands_on_the_note_onset():
    ch = reference_aligned_cues(_cues(with_voicing=False), _FakeItem(note_onset=FRAME_TIMES[5]))
    _, onset_true = ch.onset
    # The impulse (widened by one frame) is centred on frame 5, and nowhere else.
    assert onset_true[5] == 1.0
    assert onset_true[[0, 1, 9, 10, 11]].sum() == 0.0


def test_voicing_dropped_without_a_head():
    assert reference_aligned_cues(_cues(with_voicing=False), _FakeItem(0.03)).voicing is None


def test_voicing_and_pitch_aligned_to_the_frame_grid():
    ch = reference_aligned_cues(_cues(with_voicing=True), _FakeItem(0.03))
    voi_pred, voi_true = ch.voicing
    assert voi_pred.shape == voi_true.shape == FRAME_TIMES.shape
    est_hz, ref_hz = ch.pitch
    assert est_hz.shape == ref_hz.shape  # both on the reference grid
    assert np.allclose(ref_hz, 110.0)
