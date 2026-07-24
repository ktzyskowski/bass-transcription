"""PitchAccuracy -- RPA/RCA/Overall and the voicing measures.

Handcrafted with clean octave relationships so the expected numbers are exact:
110 Hz is A2, 220 Hz is A3 -- one octave apart, same pitch class. That lets the
octave-error case assert RPA=0 / RCA=1 without any cents arithmetic, and pins the
one property the whole octave story rests on: the RPA/RCA gap *is* the octave
rate.

Also pins the subtler contract that RPA/RCA are scored over reference-*voiced*
frames only (a wrong pitch during a rest cannot lower them -- only Overall sees
it), which is exactly why voicing is reported alongside.
"""

import math

import numpy as np
import pytest

from stemify.metrics.frame.pitch import PitchAccuracy

A2 = 110.0  # MIDI 45
A3 = 220.0  # MIDI 57, one octave above A2


def scores(true, pred):
    return PitchAccuracy().calculate(
        np.array(true, dtype=float), np.array(pred, dtype=float)
    ).scalars


def test_perfect_match():
    s = scores([A2, A2, A3], [A2, A2, A3])
    assert s["rpa"] == 1.0
    assert s["rca"] == 1.0
    assert s["overall"] == 1.0
    assert s["octave_gap"] == 0.0
    assert s["voicing_recall"] == 1.0
    assert s["voicing_false_alarm"] == 0.0
    assert s["n_frames"] == 3.0
    assert s["n_voiced"] == 3.0


def test_octave_error_is_rpa_zero_rca_one():
    # Every frame answered an octave high: exactly wrong (RPA), exactly the right
    # pitch class (RCA). The gap between them is the octave-error rate = 1.0.
    s = scores([A2, A2], [A3, A3])
    assert s["rpa"] == 0.0
    assert s["rca"] == 1.0
    assert s["octave_gap"] == 1.0
    # Folding the octave out leaves a perfectly in-tune note -- 0 cents error.
    assert s["median_cents"] == 0.0


def test_rpa_ignores_rest_frames_but_overall_does_not():
    # Frame 2 is a reference rest; the estimate hallucinates a pitch there.
    # RPA/RCA are over voiced frames only, so they stay perfect; Overall counts
    # the bad rest, and it registers as a voicing false alarm.
    s = scores([A2, 0.0], [A2, 999.0])
    assert s["rpa"] == 1.0
    assert s["overall"] == 0.5
    assert s["voicing_false_alarm"] == 1.0
    assert s["n_voiced"] == 1.0


def test_declining_to_answer_lowers_recall_and_rpa():
    # Estimate goes silent on a voiced reference frame: missed, so both voicing
    # recall and RPA take the hit (an unvoiced estimate is not a correct pitch).
    s = scores([A2, A2], [A2, 0.0])
    assert s["rpa"] == 0.5
    assert s["voicing_recall"] == 0.5


def test_all_rest_reference_has_no_voiced_frames():
    s = scores([0.0, 0.0], [0.0, 0.0])
    assert s["n_voiced"] == 0.0
    # No frames both sides call voiced -> tuning error is undefined, not 0.
    assert math.isnan(s["median_cents"])


def test_misaligned_arrays_are_rejected():
    with pytest.raises(ValueError):
        PitchAccuracy().calculate(np.array([A2, A2]), np.array([A2]))


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        PitchAccuracy().calculate(np.array([[A2]]), np.array([[A2]]))