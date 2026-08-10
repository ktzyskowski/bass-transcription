"""Target builders -- pure array math, so every case is hand-checkable.

The frame grid here is the real 11.6 ms CQT hop, and the source grids are the
10 ms reference grid, so these also exercise the cross-grid resample that the
whole scaffolding depends on being correct.
"""

import numpy as np

from stemify.music.contour import Contour
from stemify.music.labels import (
    build_targets, onset_targets, pitch_targets, voicing_targets,
)
from stemify.music.notes import NoteEvent

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


# --------------------------------------------------------------------------- #
# build_targets: the per-head dispatch                                        #
# --------------------------------------------------------------------------- #
def test_build_targets_aligns_onset_and_voicing():
    """Was `examples.build_example`, which returned a `CachedItem` -- so checking target
    alignment meant naming a persistence type. Arrays in, arrays out now."""
    frame_times = np.arange(20) * 0.0116
    ref_times = np.arange(24) * 0.01
    freqs = np.where(np.arange(24) < 12, 110.0, 0.0)  # voiced for the first ~0.12 s
    reference = Contour(ref_times, freqs)
    notes = [NoteEvent(onset=0.0, offset=0.12, midi_pitch=45.0)]

    targets = build_targets(frame_times, reference, notes, heads=["onset", "voicing"])

    assert set(targets) == {"onset", "voicing"}
    assert targets["onset"].shape == (20,)
    assert targets["onset"][:3].any()            # onset marked near t=0
    assert targets["voicing"][:5].all()          # voiced early
    assert not targets["voicing"][-3:].any()     # rest late


def test_only_the_requested_heads_are_built():
    """A run that trains onset alone must not pay a pitch conversion per track, and a
    cache entry must not gain a target nobody asked for -- the head set is what makes
    an entry reusable across runs."""
    reference = Contour(np.arange(10) * 0.01, np.full(10, 110.0))
    got = build_targets(np.arange(8) * 0.0116, reference, [], heads=["onset"])
    assert set(got) == {"onset"}


def test_an_unknown_head_yields_no_entry_rather_than_raising():
    """Deliberate: the caller's cache then reports it missing, which is where a wrong
    head name is actually visible -- at the point something tries to train on it."""
    reference = Contour(np.arange(10) * 0.01, np.full(10, 110.0))
    assert build_targets(np.arange(8) * 0.0116, reference, [], heads=["nonsense"]) == {}


def test_pitch_targets_carry_midi_not_hz():
    """A-110 is MIDI 45. The conversion is written out rather than taken from librosa;
    this is what pins it to the definition of the MIDI scale."""
    reference = Contour(np.arange(10) * 0.01, np.full(10, 110.0))
    got = build_targets(np.arange(8) * 0.0116, reference, [], heads=["pitch"])["pitch"]
    assert np.allclose(got[~np.isnan(got)], 45.0)
