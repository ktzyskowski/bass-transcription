"""The cache keys, pinned to literal digests. Do not "fix" a failure here by updating them.

There are 136 GB of derived artifacts on the training box -- 8158 cached CQTs and 796
pyin contours, each one a separation and a transform away from cheap. Every one of them
is addressed by ``content_key``, and **nothing in the layout records how it was keyed**.
So a change to the key does not corrupt anything and does not raise: it silently misses
on every entry, and the next run recomputes the corpus from scratch. Hours, reported as
a cold cache.

That makes the digest a compatibility surface with no version field, which is the same
shape as a wire format. These tests are the version field.

The literals below were captured from the implementation that wrote the existing cache
(``core.cache.Cache``, deleted when persistence was inverted). They were verified at the
time against 300 real corpus files -- 300/300 paths identical, 71 landing on entries
already on disk -- so a failure here means the current code would no longer find that
cache.

**If one fails**, the question is not "what is the new digest" but "what did I change".
The inputs are: ``DIGEST_CHARS``, the NUL join, the SHA-256, the ``str()`` of each part,
the producer identity, and ``Path.resolve()``. Changing any of them deliberately means
re-warming the corpus, and that is a decision to make on purpose rather than discover.
"""

from pathlib import Path

from stemify.persistence.store import DIGEST_CHARS, Store, content_key


def test_the_digest_is_what_it_has_always_been():
    """One literal, over parts chosen to exercise the join rather than any real key."""
    assert content_key("producer", "/audio/track.wav") == "1d2d12424afed153"


def test_the_digest_length_is_pinned():
    """Truncation is part of the key. Widening it invalidates all 8158 entries as
    surely as changing the hash would."""
    assert DIGEST_CHARS == 16
    assert len(content_key("x")) == 16


def test_a_feature_location_matches_the_layout_that_wrote_the_cache():
    """``<root>/features/<stem>-<digest>.npz``, digest over (SIGNATURE, resolved path).

    Spelled out as a literal rather than recomputed, so this fails if the *composition*
    changes -- a reordered ``content_key`` call, a different subdirectory, a lost stem --
    and not merely if the hash does.
    """
    store = Store(Path("/cache"))
    got = store.location("features", "cqt-sig", "/audio/Track00042.flac")
    assert got == Path(
        f"/cache/features/Track00042-{content_key('cqt-sig', '/audio/Track00042.flac')}.npz"
    )


def test_a_contour_location_matches_the_layout_that_wrote_the_cache():
    store = Store(Path("/cache"))
    got = store.location("contours", "pyin", "/audio/Track00042.flac")
    assert got == Path(
        f"/cache/contours/Track00042-{content_key('pyin', '/audio/Track00042.flac')}.npz"
    )


def test_the_source_path_is_resolved_before_hashing():
    """Two spellings of one file must be one entry. ``resolve()`` is inside the key, so
    a caller passing a relative path and one passing an absolute path agree -- which is
    what lets the validator reuse the loader's features without coordinating."""
    store = Store(Path("/cache"))
    absolute = Path("data/raw/x.wav").resolve()
    assert store.location("features", "s", "data/raw/x.wav") == store.location(
        "features", "s", absolute
    )
