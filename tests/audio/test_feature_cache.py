"""The memoised CQT: computed once per resolved audio file, then read from disk.

The behaviour worth protecting is not that caching works -- it is *what a hit is allowed
to cost*. A hit must not load audio, and it must not run the transform, because the
transform is Demucs and the whole corpus is thousands of tracks.
"""

import numpy as np
import pytest

from stemify.audio import cache as feature_cache
from stemify.audio.cache import ensure_feature, ensure_features, feature_path
from stemify.audio.features import FrameFeatures
from stemify.audio.transforms import AudioTransform, Identity
from stemify.persistence.serde import CachedItem
from stemify.persistence.store import Store


class _FakeSeparator(AudioTransform):
    """Resolves to a file distinct from its source, the way Demucs does, without
    needing Demucs."""

    @property
    def name(self) -> str:
        return "fake-sep"

    def _run(self, source, dest) -> None:
        dest.write_text("separated")


@pytest.fixture
def cqt_calls(monkeypatch):
    """Counts CQT computations and returns a fixed tiny feature array."""
    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        return FrameFeatures(
            np.random.rand(4, 10).astype(np.float32), np.arange(10) * 0.0116
        )

    monkeypatch.setattr(feature_cache, "extract_features", fake)
    return calls


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "trk.wav"
    path.write_text("x")
    return path


def test_the_cqt_is_computed_once_then_served_from_disk(tmp_path, audio, cqt_calls):
    store = Store(tmp_path / "cache")
    first = ensure_feature(store, Identity(), audio)
    assert cqt_calls["n"] == 1
    assert ensure_feature(store, Identity(), audio) == first
    assert cqt_calls["n"] == 1


def test_the_cached_entry_is_target_less(tmp_path, audio, cqt_calls):
    """Features and the frame grid, no labels. That separation is what lets one entry
    serve an onset-only run and a later run that adds voicing, without recomputing a
    single CQT -- the targets are layered on by whoever holds the reference."""
    loaded = CachedItem.load(ensure_feature(Store(tmp_path / "c"), Identity(), audio))
    assert loaded.targets == {}
    assert loaded.features.shape == (4, 10)


def test_two_conditions_of_one_track_do_not_collide(tmp_path, audio, cqt_calls):
    """What replaced the old ``cache_dir/<split>/<tag>`` partitioning. A stem and its
    separated version are the same *track* but different resolved files, so the key
    separates them and there is no subdirectory scheme left to maintain."""
    store = Store(tmp_path / "cache")
    sep = _FakeSeparator(cache_dir=store.dir("transforms"))
    assert ensure_feature(store, Identity(), audio) != ensure_feature(store, sep, audio)
    assert cqt_calls["n"] == 2
    assert len(list(store.dir("features").glob("*.npz"))) == 2


def test_a_hit_never_runs_the_transform(tmp_path, audio, cqt_calls):
    """Why the key comes from ``transform.cache_path`` and not from ``apply``.

    Asking whether a CQT is cached must not separate the track to find out. Over a warm
    cache ``_run`` is unreachable -- otherwise every hit pays minutes of Demucs, and a
    resumed run costs the same as a cold one.
    """
    store = Store(tmp_path / "cache")
    ensure_feature(store, _FakeSeparator(cache_dir=store.dir("transforms")), audio)

    class _Exploding(_FakeSeparator):
        def _run(self, source, dest):
            raise AssertionError("a cache hit must not run the transform")

    ensure_feature(store, _Exploding(cache_dir=store.dir("transforms")), audio)
    assert cqt_calls["n"] == 1


def test_the_path_is_answerable_without_computing_anything(tmp_path, audio, cqt_calls):
    """``feature_path`` is the question, ``ensure_feature`` is the work. Callers that
    already hold a warm cache (the validator reusing the loader's features) read the
    first and never reach the second."""
    store = Store(tmp_path / "cache")
    assert not feature_path(store, audio).exists()
    assert cqt_calls["n"] == 0


def test_a_sweep_returns_one_path_per_source_in_order(tmp_path, cqt_calls):
    """The sweep is a ``for`` loop over files, not over a corpus. That is the whole
    inversion: turning a ``Dataset`` into audio files is the caller's job, so this layer
    never imports one."""
    sources = []
    for i in range(3):
        p = tmp_path / f"t{i}.wav"
        p.write_text("x")
        sources.append(p)
    paths = ensure_features(Store(tmp_path / "cache"), Identity(), sources)
    assert len(paths) == len(set(paths)) == 3
    assert cqt_calls["n"] == 3
