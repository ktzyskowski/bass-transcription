"""Caching policy: keys, layout, and compute-once-then-read-from-disk behaviour.

``Cache`` is exercised with counting fakes so "computed once, then served from disk"
is asserted directly, no audio and no pyin. The on-disk formats themselves live in
test_serde.py.

The property under test is content-addressing on the *resolved* audio path. That is
what retired the old ``cache_dir/<split>/<tag>`` partitioning, so "two conditions of
one track cannot collide" is now load-bearing rather than incidental.
"""

import numpy as np

from stemify.core.cache import Cache, content_key
from stemify.core.contour import Contour
from stemify.core.notes import NoteEvent
from stemify.core.serde import CachedItem
from stemify.core.sources import Sources
from stemify.core.transforms import AudioTransform, Identity


class _FakeSeparator(AudioTransform):
    """Resolves to a file distinct from its source, the way Demucs does, without
    needing Demucs."""

    @property
    def name(self) -> str:
        return "fake-sep"

    def _run(self, source, dest) -> None:
        dest.write_text("separated")


# --------------------------------------------------------------------------- #
# keying                                                                      #
# --------------------------------------------------------------------------- #
def test_content_key_is_stable_and_separator_safe():
    assert content_key("a", "b") == content_key("a", "b")
    # NUL-joined, so two different part tuples cannot collide by concatenation.
    assert content_key("a", "b") != content_key("ab", "")


def test_feature_path_keys_on_resolved_audio(tmp_path):
    cache = Cache(tmp_path)
    a, b = tmp_path / "stem.wav", tmp_path / "sep.wav"
    assert cache.feature_path(a) == cache.feature_path(a)
    assert cache.feature_path(a) != cache.feature_path(b)
    assert cache.feature_path(a).parent == cache.features_dir


def test_feature_key_changes_with_the_cqt_signature(tmp_path, monkeypatch):
    # The whole point of the CQT signature being in the key: change a CQT constant and
    # every cached feature must miss, rather than a model trained on the new settings
    # quietly reading features computed under the old ones.
    cache = Cache(tmp_path)
    before = cache.feature_path(tmp_path / "trk.wav")
    monkeypatch.setattr("stemify.core.cache.CQT_SIGNATURE", "cqt-different-hop")
    assert cache.feature_path(tmp_path / "trk.wav") != before


# --------------------------------------------------------------------------- #
# ensure_features: the CQT computed once per resolved audio file              #
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

    def sources(self):
        # Counted like audio(): ensure_features resolves through the role, so a
        # cache hit must not reach the file this way either.
        return Sources(bass=self.audio())

    def reference(self):
        raise AssertionError("ensure_features must not build references")

    def reference_notes(self):
        return [NoteEvent(onset=0.1, offset=0.3, midi_pitch=45.0)]


def _fake_extract_features(monkeypatch, calls):
    from stemify.core import features as features_mod

    def fake(path):
        calls["n"] += 1
        t = np.arange(10) * 0.0116
        return features_mod.FrameFeatures(np.random.rand(4, 10).astype(np.float32), t)

    monkeypatch.setattr("stemify.core.cache.extract_features", fake)


def test_ensure_features_computes_once_then_skips(tmp_path, monkeypatch):
    calls = {"n": 0}
    _fake_extract_features(monkeypatch, calls)
    wav = tmp_path / "trk.wav"
    wav.write_text("x")
    cache = Cache(tmp_path / "cache")

    first = cache.ensure_features([_CountingItem(wav)], Identity())
    assert calls["n"] == 1
    second = cache.ensure_features([_CountingItem(wav)], Identity())
    assert calls["n"] == 1  # hit: no CQT recompute
    assert first == second
    loaded = CachedItem.load(first[0])
    assert loaded.targets == {}  # target-less: features only
    assert loaded.features.shape == (4, 10)


def test_two_conditions_of_one_track_do_not_collide(tmp_path, monkeypatch):
    # What replaces the cache_dir/<tag> partitioning: a stem and its separated version
    # are the same *track* but different resolved files, so the key separates them
    # with no subdirectory scheme left to maintain.
    calls = {"n": 0}
    _fake_extract_features(monkeypatch, calls)
    wav = tmp_path / "trk.wav"
    wav.write_text("x")
    cache = Cache(tmp_path / "cache")
    sep = _FakeSeparator(cache_dir=cache.transforms_dir)

    stem_paths = cache.ensure_features([_CountingItem(wav)], Identity())
    sep_paths = cache.ensure_features([_CountingItem(wav)], sep)
    assert stem_paths != sep_paths
    assert calls["n"] == 2
    assert len(list(cache.features_dir.glob("*.npz"))) == 2


def test_a_cache_hit_never_runs_the_transform(tmp_path, monkeypatch):
    # Why keys come from `transform.cache_path` and not `apply`: asking whether a CQT
    # is cached must not separate the track first. Over a warm cache, _run must never
    # be reached -- otherwise every hit would pay minutes of Demucs.
    calls = {"n": 0}
    _fake_extract_features(monkeypatch, calls)
    wav = tmp_path / "trk.wav"
    wav.write_text("x")
    cache = Cache(tmp_path / "cache")

    class _ExplodingSeparator(_FakeSeparator):
        def _run(self, source, dest):
            raise AssertionError("cache hit must not run the transform")

    cache.ensure_features(
        [_CountingItem(wav)], _FakeSeparator(cache_dir=cache.transforms_dir)
    )
    cache.ensure_features(
        [_CountingItem(wav)], _ExplodingSeparator(cache_dir=cache.transforms_dir)
    )
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# contour: the pyin contour computed once per resolved audio path             #
# --------------------------------------------------------------------------- #
class _CountingEstimator:
    """Stands in for a PitchEstimator: returns a fixed contour, counts calls."""

    name = "pyin"

    def __init__(self):
        self.calls = 0

    def estimate(self, path):
        self.calls += 1
        times = np.linspace(0, 1, 5)
        return Contour(times=times, freqs=np.full(5, 110.0), confidence=np.full(5, 0.9))


def test_contour_computes_once_then_reads_disk(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_text("x")
    cache = Cache(tmp_path)
    est = _CountingEstimator()

    first = cache.contour(est, audio)
    assert est.calls == 1
    second = cache.contour(est, audio)  # served from disk
    assert est.calls == 1
    np.testing.assert_array_equal(first.freqs, second.freqs)
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert list(cache.contours_dir.glob("*.partial.npz")) == []  # no temp left behind


def test_contour_keys_on_path_so_conditions_dont_collide(tmp_path):
    a, b = tmp_path / "stem.wav", tmp_path / "demucs.wav"
    a.write_text("a")
    b.write_text("b")
    cache = Cache(tmp_path)
    est = _CountingEstimator()
    cache.contour(est, a)
    cache.contour(est, b)
    assert est.calls == 2  # no collision -> both computed
    assert len(list(cache.contours_dir.glob("*.npz"))) == 2


# --------------------------------------------------------------------------- #
# layout                                                                      #
# --------------------------------------------------------------------------- #
def test_layout_is_owned_by_the_cache(tmp_path):
    # Callers ask the cache where things live rather than assembling paths, which is
    # what let train.py stop hand-building five subdirectories.
    cache = Cache(tmp_path)
    assert cache.features_dir == tmp_path / "features"
    assert cache.contours_dir == tmp_path / "contours"
    assert cache.transforms_dir == tmp_path / "transforms"


def test_separator_writes_into_this_cache(tmp_path):
    cache = Cache(tmp_path)
    sep = cache.separator(model="htdemucs", stem="bass")
    assert sep.cache_dir == cache.transforms_dir
    assert sep.name == "demucs-htdemucs-bass"
