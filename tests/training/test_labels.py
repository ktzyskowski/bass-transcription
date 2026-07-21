"""Target builders -- pure array math, so every case is hand-checkable.

The frame grid here is the real 11.6 ms CQT hop, and the source grids are the
10 ms reference grid, so these also exercise the cross-grid resample that the
whole scaffolding depends on being correct.
"""

import numpy as np

from stemify.training.labels import onset_targets, pitch_targets, voicing_targets

FRAMES = np.arange(10) * 0.0116  # 11.6 ms CQT grid
REF = np.arange(12) * 0.01       # 10 ms reference grid


def test_onset_lands_on_nearest_frame_and_widens():
    # Onset near frame 3, widened by one frame each side.
    y = onset_targets(np.array([0.0348]), FRAMES, tolerance_frames=1)
    assert list(np.flatnonzero(y)) == [2, 3, 4]


def test_onset_tolerance_zero_is_a_single_frame():
    y = onset_targets(np.array([0.0348]), FRAMES, tolerance_frames=0)
    assert list(np.flatnonzero(y)) == [3]


def test_onset_empty_is_all_zero():
    y = onset_targets(np.array([]), FRAMES)
    assert y.shape == FRAMES.shape
    assert not y.any()


def test_onsets_near_the_edges_are_clipped_not_wrapped():
    y = onset_targets(np.array([0.0, FRAMES[-1]]), FRAMES, tolerance_frames=1)
    assert y[0] == 1.0 and y[1] == 1.0  # widened right only at the start
    assert y[-1] == 1.0 and y[-2] == 1.0  # widened left only at the end


def test_voicing_all_voiced_and_all_rest():
    voiced = np.ones(REF.size, dtype=bool)
    assert voicing_targets(REF, voiced, FRAMES).all()
    assert not voicing_targets(REF, ~voiced, FRAMES).any()


def test_voicing_resamples_a_step_monotonically():
    # First half voiced, second half rest, on the 10 ms grid.
    mask = np.array([True] * 6 + [False] * 6)
    y = voicing_targets(REF, mask, FRAMES)
    # Resampled to 11.6 ms: an unbroken run of 1s then 0s, no interleaving.
    ones = np.flatnonzero(y)
    assert list(ones) == list(range(ones.size))  # a contiguous prefix


def test_pitch_carries_voiced_midi_and_nans_rests():
    midi = np.array([40.0, 40.0] + [np.nan] * 10)
    y = pitch_targets(REF, midi, FRAMES)
    assert y[0] == 40.0 and y[1] == 40.0
    assert np.isnan(y[5:]).all()  # deep in the rest, no pitch invented