"""Base AudioTransform caching: the atomic write-aside-and-rename path in
``apply()``, exercised with a fake transform so no real (heavy) separator runs.
This is the path that a missing ``import os`` silently broke -- every cache miss
raised NameError on ``os.getpid()`` while cache hits sailed through untouched.
"""

from pathlib import Path

from stemify.core.transforms import AudioTransform, Identity


class _FakeTransform(AudioTransform):
    """Writes a marker file instead of separating; counts how often it ran."""

    @property
    def name(self) -> str:
        return "fake"

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.runs = 0

    def _run(self, source: Path, dest: Path) -> None:
        self.runs += 1
        dest.write_text(f"transformed:{Path(source).name}")


def _source(tmp_path: Path) -> Path:
    src = tmp_path / "in.wav"
    src.write_text("audio")
    return src


def test_apply_computes_then_caches(tmp_path):
    # First call runs the transform (the cache-miss path with os.getpid());
    # second call is a pure cache hit and must not re-run.
    t = _FakeTransform(tmp_path / "cache")
    src = _source(tmp_path)

    out = t.apply(src)
    assert out.exists() and out.read_text() == "transformed:in.wav"
    assert t.runs == 1

    again = t.apply(src)
    assert again == out
    assert t.runs == 1  # served from cache, not recomputed


def test_apply_leaves_no_partial_files(tmp_path):
    t = _FakeTransform(tmp_path / "cache")
    t.apply(_source(tmp_path))
    partials = list((tmp_path / "cache").rglob("*.partial"))
    assert partials == []  # temp file renamed away, none left behind


def test_failed_run_cleans_up_and_writes_no_cache(tmp_path):
    class _Boom(_FakeTransform):
        def _run(self, source, dest):
            dest.write_text("half")  # a truncated write...
            raise RuntimeError("separator died")  # ...before completion

    t = _Boom(tmp_path / "cache")
    src = _source(tmp_path)
    try:
        t.apply(src)
    except RuntimeError:
        pass
    # No poisoned cache entry, no leftover partial masquerading as a hit.
    assert not t.cache_path(src).exists()
    assert list((tmp_path / "cache").rglob("*.partial")) == []


def test_identity_returns_source_untouched(tmp_path):
    src = _source(tmp_path)
    assert Identity().apply(src) == src


def test_identity_cache_path_is_the_source(tmp_path):
    # cache_path doubles as the resolved-audio key every derived artifact is keyed on
    # (stemify.core.cache), so Identity must report the *source*. The inherited digest path
    # would name a .flac that is never written, and every cached feature for an
    # untransformed track would then address a nonexistent file.
    src = _source(tmp_path)
    identity = Identity(cache_dir=tmp_path / "cache")
    assert identity.cache_path(src) == src
    assert identity.cache_path(src) == identity.apply(src)


def test_cache_path_does_no_work(tmp_path):
    # The property Cache.ensure_features relies on to avoid separating a track just to
    # discover its CQT is already cached: resolving the path must never call _run.
    class _Exploding(AudioTransform):
        @property
        def name(self):
            return "exploding"

        def _run(self, source, dest):
            raise AssertionError("cache_path must not run the transform")

    src = _source(tmp_path)
    resolved = _Exploding(cache_dir=tmp_path / "cache").cache_path(src)
    assert resolved.suffix == ".flac" and not resolved.exists()
