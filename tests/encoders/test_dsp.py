"""ClassicalEncoder: the pitch source is injected, not hardcoded.

The whole point of the pitch/ split -- the encoder holds a PitchEstimator, so a
learned head drops in with no change here. Exercised with a fake estimator and
onsets off, so no audio is read.
"""

import numpy as np

from stemify.core.contour import Contour
from stemify.core.encoders.classical import ClassicalEncoder
from stemify.core.pitch import PitchEstimator, PyinPitch
from stemify.audio.sources import Sources


class _FakePitch(PitchEstimator):
    name = "fake"

    def estimate(self, audio_path) -> Contour:
        return Contour(times=np.arange(5) * 0.01, freqs=np.full(5, 123.0))


def test_encode_uses_the_injected_estimator():
    cues = ClassicalEncoder(pitch=_FakePitch(), use_onsets=False).encode(Sources(bass="unused"))
    np.testing.assert_array_equal(cues.contour.freqs, np.full(5, 123.0))
    assert cues.onset.shape == cues.frame_times.shape
    assert cues.voicing is None  # DSP floor has no voicing head


def test_default_estimator_is_pyin():
    assert isinstance(ClassicalEncoder().pitch, PyinPitch)
