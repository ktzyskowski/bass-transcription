"""Caching policy: cache keys, and compute-once-then-read-from-disk behaviour.

cached_f0 and cache_features are exercised with counting fakes so "computed once,
then served from disk" is asserted directly, no audio or pyin. The on-disk formats
themselves live in test_serde.py.
"""

import numpy as np

from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.serde import TrackExample
from stemify.training.cache import cache_features, cached_f0, example_path
from stemify.transforms import Identity


def test_example_path_keys_on_name_and_transform(tmp_path):
    # The writer (cache_features) and every reader must agree on this filename, so
    # it lives in one helper. '/' in a track name is flattened, and the transform is
    # part of the key so stem and demucs of one track don't collide.
    assert example_path(tmp_path, "Track01", "identity").name == "Track01__identity.npz"
    assert example_path(tmp_path, "a/b", "demucs").name == "a_b__demucs.npz"
    stem = example_path(tmp_path, "Track01", "identity")
    demucs = example_path(tmp_path, "Track01", "demucs-htdemucs-bass")
    assert stem != demucs  # same track, different transform -> distinct files


# --------------------------------------------------------------------------- #
# cache_features: the CQT computed once per (track, transform)                #
# --------------------------------------------------------------------------- #
class _CountingItem:
    """A dataset item whose audio() call is counted -- a cache hit must not load it."""

    def __init__(self, path):
        self._path = path
        self.audio_calls = 0

    @property
    def name(self):
        return self._path.stem

    def audio(self):
        self.audio_calls += 1
        return self._path

    def reference(self):
        raise AssertionError("cache_features must not build references")

    def reference_notes(self):
        return [NoteEvent(onset=0.1, offset=0.3, midi_pitch=45.0)]


def test_cache_features_computes_once_then_skips(tmp_path, monkeypatch):
    from stemify import features as features_mod
    from stemify.training import cache as cache_mod

    calls = {"n": 0}

    def fake_extract(path):
        calls["n"] += 1
        t = np.arange(10) * 0.0116
        return features_mod.Features(np.random.rand(4, 10).astype(np.float32), t)

    monkeypatch.setattr(cache_mod, "extract_features", fake_extract)
    wav = tmp_path / "trk.wav"
    wav.write_text("x")
    item = _CountingItem(wav)

    first = cache_features([item], tmp_path / "cache", Identity())
    assert calls["n"] == 1 and item.audio_calls == 1
    second = cache_features([item], tmp_path / "cache", Identity())
    assert calls["n"] == 1 and item.audio_calls == 1   # hit: no audio, no CQT
    assert first == second
    loaded = TrackExample.load(first[0])
    assert loaded.targets == {}                        # target-less: features only
    assert loaded.features.shape == (4, 10)


# --------------------------------------------------------------------------- #
# cached_f0: the pyin contour computed once per resolved audio path           #
# --------------------------------------------------------------------------- #
class _CountingBaseline:
    """Stands in for a PitchEstimator: returns a fixed contour, counts calls."""

    name = "pyin"

    def __init__(self):
        self.calls = 0

    def estimate(self, path):
        self.calls += 1
        times = np.linspace(0, 1, 5)
        return Contour(times=times, freqs=np.full(5, 110.0),
                       confidence=np.full(5, 0.9))


def test_cached_f0_computes_once_then_reads_disk(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_text("x")
    cache = tmp_path / "contours"
    base = _CountingBaseline()

    first = cached_f0(base, audio, cache)
    assert base.calls == 1
    second = cached_f0(base, audio, cache)  # served from disk
    assert base.calls == 1
    np.testing.assert_array_equal(first.freqs, second.freqs)
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert list(cache.glob("*.partial.npz")) == []  # atomic write left no temp


def test_cached_f0_keys_on_path_so_conditions_dont_collide(tmp_path):
    # Two different audio files (stem vs separated) must get distinct cache entries.
    a, b = tmp_path / "stem.wav", tmp_path / "demucs.wav"
    a.write_text("a")
    b.write_text("b")
    cache = tmp_path / "contours"
    base = _CountingBaseline()
    cached_f0(base, a, cache)
    cached_f0(base, b, cache)
    assert base.calls == 2  # no collision -> both computed
    assert len(list(cache.glob("*.npz"))) == 2


def test_cached_f0_none_disables_cache(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_text("x")
    base = _CountingBaseline()
    cached_f0(base, audio, None)
    cached_f0(base, audio, None)
    assert base.calls == 2  # recomputed each time, nothing written
