"""PitchConfusion -- the semitone-delta histogram and its summary rates.

Same clean pitches as the accuracy tests (110 Hz = MIDI 45, 220 Hz = MIDI 57),
so deltas come out as exact integers. The cases pin the three things the metric
is for: catching +12 octave locking, keeping out-of-range harmonics visible in
the histogram while excluding them from the fixed-axis matrix, and scoring over
reference-voiced frames with unanswered frames counted (not dropped).
"""

import numpy as np
import pytest

from stemify.core.metrics.frame.confusion import MIDI_MAX, MIDI_MIN, PitchConfusion

A2 = 110.0   # MIDI 45
A3 = 220.0   # MIDI 57  (+12 from A2)
A5 = 880.0   # MIDI 81  (above MIDI_MAX=67 -> out of range)
N_CLASSES = MIDI_MAX - MIDI_MIN + 1  # 45


def result(true, pred):
    return PitchConfusion().compute(
        np.array(true, dtype=float), np.array(pred, dtype=float)
    )


def test_exact_lands_on_the_diagonal():
    r = result([A2, A2], [A2, A2])
    s = r.scalars
    assert s["exact_rate"] == 1.0
    assert s["octave_up_rate"] == 0.0
    assert list(r.arrays["delta_values"]) == [0]
    assert list(r.arrays["delta_counts"]) == [2]
    # both frames at (45, 45) in matrix coordinates
    assert r.arrays["matrix"][45 - MIDI_MIN, 45 - MIDI_MIN] == 2
    assert r.arrays["matrix"].sum() == 2


def test_octave_up_is_a_plus_twelve_spike():
    r = result([A2, A2], [A3, A3])
    s = r.scalars
    assert s["exact_rate"] == 0.0
    assert s["octave_up_rate"] == 1.0
    assert list(r.arrays["delta_values"]) == [12]
    # true 45 -> pred 57, both in range, off the diagonal
    assert r.arrays["matrix"][45 - MIDI_MIN, 57 - MIDI_MIN] == 2


def test_out_of_range_stays_in_histogram_but_out_of_matrix():
    # A harmonic above the 5-string range: it must still show in the delta
    # histogram (that is the failure we want to see) but not enter the matrix.
    r = result([A2], [A5])
    s = r.scalars
    assert s["out_of_range_rate"] == 1.0
    assert list(r.arrays["delta_values"]) == [36]  # 81 - 45
    assert r.arrays["matrix"].sum() == 0


def test_unanswered_voiced_frame_counts_in_denominator():
    # Estimate goes silent on one of two voiced reference frames. The unvoiced
    # frame is not in the histogram, but exact_rate's denominator is voiced
    # frames (2), not answered frames (1) -> 1/2.
    r = result([A2, A2], [A2, 0.0])
    s = r.scalars
    assert s["unvoiced_rate"] == 0.5
    assert s["exact_rate"] == 0.5
    assert list(r.arrays["delta_values"]) == [0]
    assert list(r.arrays["delta_counts"]) == [1]


def test_matrix_axes_are_fixed_size():
    r = result([A2], [A2])
    assert r.arrays["matrix"].shape == (N_CLASSES, N_CLASSES)
    assert list(r.arrays["midi_labels"]) == list(range(MIDI_MIN, MIDI_MAX + 1))


def test_misaligned_arrays_are_rejected():
    with pytest.raises(ValueError):
        PitchConfusion().compute(np.array([A2, A2]), np.array([A2]))
