"""HeuristicSegmenter.group -- the pure contour-to-notes core.

Exercised without audio by passing hand-picked onsets, so the boundary logic and
the same-pitch merge are tested in isolation. The merge cases use a generous gap
(0.05 s) so they assert the *logic*, not a float boundary.
"""

import numpy as np

from stemify.contour import Contour
from stemify.segmentation import HeuristicSegmenter

HOP = 0.01
A2_HZ = 110.0  # MIDI 45
A3_HZ = 220.0  # MIDI 57


def contour(freqs: list[float]) -> Contour:
    times = np.arange(len(freqs)) * HOP
    return Contour(times, np.array(freqs, dtype=float))


def test_onset_splits_a_held_note_into_two():
    # A single held pitch with an onset in the middle -> two same-pitch notes.
    held = contour([A2_HZ] * 100)
    seg = HeuristicSegmenter(merge_gap_s=None)
    notes = seg.group(held, onsets=np.array([0.5]))
    assert len(notes) == 2
    assert all(abs(n.midi_pitch - 45.0) < 0.5 for n in notes)


def test_merge_rejoins_an_onset_split():
    held = contour([A2_HZ] * 100)
    seg = HeuristicSegmenter(merge_gap_s=0.05)
    notes = seg.group(held, onsets=np.array([0.5]))
    assert len(notes) == 1
    assert abs(notes[0].midi_pitch - 45.0) < 0.5


def test_merge_rejoins_a_one_frame_dropout():
    # pyin drops a single frame mid-note; without merge that is two notes.
    freqs = [A2_HZ] * 100
    freqs[50] = 0.0  # one unvoiced frame
    dropout = contour(freqs)
    assert len(HeuristicSegmenter(merge_gap_s=None).group(dropout)) == 2
    assert len(HeuristicSegmenter(merge_gap_s=0.05).group(dropout)) == 1


def test_merge_keeps_distinct_pitches_apart():
    # Half at A2, half at A3: a real pitch change, must stay two notes even with
    # merge on (they are not the same pitch).
    step = contour([A2_HZ] * 50 + [A3_HZ] * 50)
    notes = HeuristicSegmenter(merge_gap_s=0.05).group(step)
    assert len(notes) == 2
    assert abs(notes[0].midi_pitch - 45.0) < 0.5
    assert abs(notes[1].midi_pitch - 57.0) < 0.5


def test_merge_keeps_notes_across_a_real_rest():
    # Same pitch, but separated by a 0.3 s rest -- a genuine repeat, gap well
    # beyond merge_gap_s, so it stays two notes.
    freqs = [A2_HZ] * 40 + [0.0] * 30 + [A2_HZ] * 30
    notes = HeuristicSegmenter(merge_gap_s=0.05).group(contour(freqs))
    assert len(notes) == 2


def test_merge_disabled_is_the_default():
    assert HeuristicSegmenter().merge_gap_s is None


def test_empty_contour_is_no_notes():
    assert HeuristicSegmenter(merge_gap_s=0.05).group(Contour.empty()) == []