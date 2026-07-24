"""``group_contour`` and ``onset_times_from_probs``: the pure decode cores.

Exercised without audio by passing hand-picked onsets/voiced masks, so the boundary
logic (split at rest/pitch-change/onset), the same-pitch merge repair, and the
peak-pick are tested in isolation. Migrated from the old ``segmentation`` tests when
the grouping moved into ``decoders.heuristic``; the merge cases use a generous gap
(0.05 s) so they assert the *logic*, not a float boundary.
"""

import numpy as np
import pytest

from stemify.contour import Contour
from stemify.decoders.heuristic import (
    MERGE_GAP_S,
    group_contour,
    onset_times_from_probs,
)

HOP = 0.01
A2_HZ = 110.0  # MIDI 45
A3_HZ = 220.0  # MIDI 57


def contour(freqs: list[float]) -> Contour:
    times = np.arange(len(freqs)) * HOP
    return Contour(times, np.array(freqs, dtype=float))


# --------------------------------------------------------------------------- #
# group_contour: split + merge                                                #
# --------------------------------------------------------------------------- #
def test_onset_splits_a_held_note_into_two():
    held = contour([A2_HZ] * 100)
    notes = group_contour(held, onsets=np.array([0.5]), merge_gap_s=None)
    assert len(notes) == 2
    assert all(abs(n.midi_pitch - 45.0) < 0.5 for n in notes)


def test_merge_rejoins_an_onset_split():
    held = contour([A2_HZ] * 100)
    notes = group_contour(held, onsets=np.array([0.5]), merge_gap_s=0.05)
    assert len(notes) == 1
    assert abs(notes[0].midi_pitch - 45.0) < 0.5


def test_merge_rejoins_a_one_frame_dropout():
    # pyin drops a single frame mid-note; without merge that is two notes.
    freqs = [A2_HZ] * 100
    freqs[50] = 0.0  # one unvoiced frame
    dropout = contour(freqs)
    assert len(group_contour(dropout, merge_gap_s=None)) == 2
    assert len(group_contour(dropout, merge_gap_s=0.05)) == 1


def test_voiced_mask_overrides_pyin_voicing():
    # pyin tracks a pitch across the whole clip (one note by default), but the
    # learned voicing mask calls the middle a rest -> two notes split by that rest.
    held = contour([A2_HZ] * 100)
    voiced = np.ones(100, dtype=bool)
    voiced[45:55] = False
    assert len(group_contour(held, voiced=np.ones(100, dtype=bool), merge_gap_s=None)) == 1
    assert len(group_contour(held, voiced=voiced, merge_gap_s=None)) == 2


def test_voiced_mask_can_trim_pyin_over_voicing():
    held = contour([A2_HZ] * 100)
    voiced = np.zeros(100, dtype=bool)
    voiced[:50] = True
    notes = group_contour(held, voiced=voiced, merge_gap_s=None)
    assert len(notes) == 1
    assert notes[0].offset < 0.6  # ends around frame 50 (0.5 s), not 1.0 s


def test_voiced_frame_without_pyin_pitch_places_no_note():
    silent = contour([0.0] * 100)          # pyin tracked nothing
    voiced = np.ones(100, dtype=bool)      # but the mask says voiced everywhere
    assert group_contour(silent, voiced=voiced, merge_gap_s=None) == []


def test_voiced_mask_shape_must_match_contour():
    held = contour([A2_HZ] * 100)
    with pytest.raises(ValueError):
        group_contour(held, voiced=np.ones(50, dtype=bool))


def test_merge_keeps_distinct_pitches_apart():
    step = contour([A2_HZ] * 50 + [A3_HZ] * 50)
    notes = group_contour(step, merge_gap_s=0.05)
    assert len(notes) == 2
    assert abs(notes[0].midi_pitch - 45.0) < 0.5
    assert abs(notes[1].midi_pitch - 57.0) < 0.5


def test_merge_keeps_notes_across_a_real_rest():
    freqs = [A2_HZ] * 40 + [0.0] * 30 + [A2_HZ] * 30
    assert len(group_contour(contour(freqs), merge_gap_s=0.05)) == 2


def test_merge_disabled_is_the_default():
    assert MERGE_GAP_S is None


def test_empty_contour_is_no_notes():
    assert group_contour(Contour.empty(), merge_gap_s=0.05) == []


# --------------------------------------------------------------------------- #
# onset_times_from_probs: peak-pick                                           #
# --------------------------------------------------------------------------- #
def test_peak_pick_finds_maxima_above_threshold():
    frame_times = np.arange(20) * 0.01
    probs = np.zeros(20)
    probs[[5, 12]] = 0.9  # two clear peaks
    probs[8] = 0.3        # below threshold
    times = onset_times_from_probs(probs, frame_times, threshold=0.5, min_gap_s=0.02)
    assert np.allclose(times, [0.05, 0.12])


def test_peak_pick_enforces_min_gap():
    frame_times = np.arange(20) * 0.01
    probs = np.zeros(20)
    probs[10] = 0.9
    probs[11] = 0.8  # adjacent, within the refractory gap -> suppressed
    times = onset_times_from_probs(probs, frame_times, threshold=0.5, min_gap_s=0.05)
    assert len(times) == 1
    assert np.isclose(times[0], 0.10)


def test_peak_pick_empty():
    assert onset_times_from_probs(np.zeros(0), np.zeros(0)).size == 0
