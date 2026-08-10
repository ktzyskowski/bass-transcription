"""The memoised pitch contour: computed once per resolved audio path, ever.

Worth more than the CQT memoiser, because a pyin contour depends on the audio and the
estimator's fixed parameters and on *nothing else* -- no checkpoint, no condition. So it
is reused across every checkpoint and every future run, which is what makes a large
validation set affordable.

The estimator is the seam. A fake ``PitchEstimator`` is a class with a name and one
method, which is the whole reason this is better than what it replaced: the same tests
used to subclass a persistence god-object to stop pyin running.
"""

import numpy as np

from stemify.transcription.pitch.cache import ensure_contour
from stemify.music.contour import Contour
from stemify.persistence.store import Store


class _CountingEstimator:
    """Returns a fixed contour and counts how often it was asked."""

    name = "pyin"

    def __init__(self, name: str = "pyin"):
        self.name = name
        self.calls = 0

    def estimate(self, path):
        self.calls += 1
        return Contour(
            times=np.linspace(0, 1, 5),
            freqs=np.full(5, 110.0),
            confidence=np.full(5, 0.9),
        )


def _audio(tmp_path, name="stem.wav"):
    path = tmp_path / name
    path.write_text("x")
    return path


def test_the_contour_is_computed_once_then_read_from_disk(tmp_path):
    store, est = Store(tmp_path), _CountingEstimator()
    audio = _audio(tmp_path)

    first = ensure_contour(store, est, audio)
    assert est.calls == 1
    second = ensure_contour(store, est, audio)
    assert est.calls == 1
    np.testing.assert_array_equal(first.freqs, second.freqs)


def test_confidence_survives_the_round_trip(tmp_path):
    """Optional on ``Contour`` -- only estimators have one -- so it is the field a
    format change drops silently, leaving a reference-shaped object with no weights."""
    store, est = Store(tmp_path), _CountingEstimator()
    audio = _audio(tmp_path)
    ensure_contour(store, est, audio)
    np.testing.assert_array_equal(ensure_contour(store, est, audio).confidence, 0.9)


def test_conditions_of_one_track_do_not_collide(tmp_path):
    """A stem and its separated version are different resolved files, so each caches
    independently. Without that, scoring the separated condition would serve the
    contour computed from the clean stem -- a better number than the model earned."""
    store, est = Store(tmp_path), _CountingEstimator()
    ensure_contour(store, est, _audio(tmp_path, "stem.wav"))
    ensure_contour(store, est, _audio(tmp_path, "demucs.wav"))
    assert est.calls == 2
    assert len(list(store.dir("contours").glob("*.npz"))) == 2


def test_two_estimators_do_not_collide(tmp_path):
    """``estimator.name`` is in the key, so pyin's contour and a learned head's are
    separate entries rather than whichever ran first."""
    store = Store(tmp_path)
    audio = _audio(tmp_path)
    ensure_contour(store, _CountingEstimator("pyin"), audio)
    ensure_contour(store, _CountingEstimator("neural-v3"), audio)
    assert len(list(store.dir("contours").glob("*.npz"))) == 2


def test_no_partial_file_is_left_behind(tmp_path):
    """Every write goes through ``atomic_write``. A truncated contour is a cache *hit*
    on the next run, so the failure is not a crash but a validation number computed
    from half a pitch track."""
    store, est = Store(tmp_path), _CountingEstimator()
    ensure_contour(store, est, _audio(tmp_path))
    assert list(store.dir("contours").glob("*.partial*")) == []
