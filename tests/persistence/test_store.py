"""``Store``: the key and the layout, which is all it does.

Everything here is a pure function of strings and paths -- no audio, no arrays, no
disk. That is the measure of the inversion: the predecessor's tests needed a counting
dataset item, a fake separator and a monkeypatched CQT to say anything at all about
keying, because keying was tangled up with a method that swept a corpus.
"""

from pathlib import Path

from stemify.persistence.store import Store, content_key


def test_the_key_is_stable():
    assert content_key("a", "b") == content_key("a", "b")


def test_the_key_is_separator_safe():
    """NUL-joined, so two different part tuples cannot collide by concatenation. The
    parts are a producer identity and a path, and both contain arbitrary text."""
    assert content_key("a", "b") != content_key("ab", "")


def test_a_location_is_stable_and_distinguishes_sources():
    store = Store(Path("/cache"))
    a = store.location("features", "cqt-v1", "/audio/stem.wav")
    b = store.location("features", "cqt-v1", "/audio/sep.wav")
    assert a == store.location("features", "cqt-v1", "/audio/stem.wav")
    assert a != b


def test_the_producer_is_in_the_key():
    """The reason a CQT constant can be changed safely. Edit the hop and ``SIGNATURE``
    changes, so every cached feature misses -- rather than a model trained on the new
    settings quietly reading arrays computed under the old ones. That failure is silent:
    a stale CQT is a valid-looking array of the right shape."""
    store = Store(Path("/cache"))
    assert store.location("features", "cqt-v1", "/a.wav") != store.location(
        "features", "cqt-different-hop", "/a.wav"
    )


def test_kinds_do_not_share_a_directory():
    store = Store(Path("/cache"))
    assert store.dir("features") == Path("/cache/features")
    assert store.dir("contours") == Path("/cache/contours")
    assert store.dir("transforms") == Path("/cache/transforms")


def test_a_location_lands_in_its_kinds_directory_and_keeps_the_stem():
    """The stem is in the filename so the tree is browsable by eye; the digest is what
    disambiguates. Both halves matter -- a digest-only name makes a cache impossible to
    inspect, and a stem-only name collides across conditions."""
    path = Store(Path("/cache")).location("contours", "pyin", "/audio/Track00042.flac")
    assert path.parent == Path("/cache/contours")
    assert path.name.startswith("Track00042-")
    assert path.suffix == ".npz"


def test_a_location_creates_nothing(tmp_path):
    """Pure. This is what lets a caller ask "is it cached?" for free -- were the path
    only obtainable by producing the artifact, the question would cost a separation per
    cache *hit*."""
    store = Store(tmp_path)
    store.location("features", "cqt-v1", tmp_path / "a.wav")
    assert list(tmp_path.iterdir()) == []
