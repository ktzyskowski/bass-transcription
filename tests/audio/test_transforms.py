"""``AudioTransform``: the atomic write-aside-and-rename in ``apply``, and where it lands.

Exercised with a fake transform so no real separator runs. This is the path a missing
``import os`` once silently broke -- every cache miss raised ``NameError`` on
``os.getpid()`` while cache hits sailed through untouched.

A transform takes the destination as a ``Store`` rather than owning a ``cache_dir``.
That is not decoration: the conditions registry builds its separators at import time, so
a transform that carried its own directory carried a *literal* one, and pointing
``--cache-dir`` elsewhere moved the features and contours while the separated audio
stayed behind.
"""

from pathlib import Path

import pytest

from stemify.audio.transforms import AudioTransform, Identity
from stemify.persistence.store import Store


class _FakeTransform(AudioTransform):
    """Writes a marker file instead of separating; counts how often it ran."""

    @property
    def name(self) -> str:
        return "fake"

    def __init__(self) -> None:
        self.runs = 0

    def _run(self, source: Path, dest: Path) -> None:
        self.runs += 1
        dest.write_text(f"transformed:{Path(source).name}")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "cache")


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "in.wav"
    src.write_text("audio")
    return src


def test_apply_computes_then_caches(store, source):
    """First call takes the cache-miss path (the one with ``os.getpid()``); the second
    is a pure hit and must not re-run -- a separation is minutes per track."""
    t = _FakeTransform()
    out = t.apply(store, source)
    assert out.read_text() == "transformed:in.wav"
    assert t.runs == 1

    assert t.apply(store, source) == out
    assert t.runs == 1


def test_the_output_lands_under_the_store(store, source):
    """The whole point of the inversion. The caller's root decides where a separation
    goes, so `--cache-dir` moves every artifact together instead of only two of three."""
    out = _FakeTransform().apply(store, source)
    assert out.parent == store.dir("transforms") / "fake"


def test_two_stores_are_two_caches(tmp_path, source):
    """Which is what could not be expressed before: the directory was fixed at
    construction, and the registry constructs its separators at import time."""
    t = _FakeTransform()
    a = t.apply(Store(tmp_path / "a"), source)
    b = t.apply(Store(tmp_path / "b"), source)
    assert a != b
    assert a.name == b.name  # same key, different roots
    assert t.runs == 2


def test_apply_leaves_no_partial_files(store, source):
    _FakeTransform().apply(store, source)
    assert list(store.root.rglob("*.partial")) == []


def test_a_failed_run_writes_no_cache_entry(store, source):
    """A truncated separation is a cache *hit* next run -- audio the model was never
    trained on, silently. Neither the destination nor a partial may survive."""

    class _Boom(_FakeTransform):
        def _run(self, src, dest):
            dest.write_text("half")
            raise RuntimeError("separator died")

    t = _Boom()
    with pytest.raises(RuntimeError, match="separator died"):
        t.apply(store, source)
    assert not t.cache_path(store, source).exists()
    assert list(store.root.rglob("*.partial")) == []


def test_identity_reports_the_source_as_its_own_output(store, source):
    """``cache_path`` doubles as the resolved-audio key every derived artifact is keyed
    on, so Identity must report the *source*. The inherited digest path would name a
    .flac that is never written, and every cached feature for an untransformed track
    would then address a file that does not exist."""
    identity = Identity()
    assert identity.cache_path(store, source) == source
    assert identity.apply(store, source) == source


def test_cache_path_does_no_work(store, source):
    """What `audio.cache` relies on to avoid separating a track just to discover its CQT
    is already cached. Resolving the path must never reach ``_run``."""

    class _Exploding(AudioTransform):
        @property
        def name(self):
            return "exploding"

        def _run(self, src, dest):
            raise AssertionError("cache_path must not run the transform")

    resolved = _Exploding().cache_path(store, source)
    assert resolved.suffix == ".flac"
    assert not resolved.exists()
