"""``Store``: where a derived artifact lives, and nothing about how it is made.

Every expensive thing this project computes -- a separated stem, a pyin contour, a CQT
-- is a pure function of some *resolved audio file* plus the identity of whatever
produced it. This owns the two decisions that follow from that and no others: the
directory layout, and the key.

**What it deliberately does not do.** This replaced a ``Cache`` class whose central
method was ``ensure_features(dataset, transform, role)`` -- one call that walked a
corpus, resolved a transform, computed a CQT, wrote an npz and returned paths. Five
concerns, so it had to import all five, and the tell was that every training test
subclassed it into a ``_NoOpCache`` just to get it out of the way. Measured before the
change: nine public members, imported by 24 files, called in three places; seven of the
nine had no production caller at all.

So the store answers *where*, callers decide *whether to compute*. The memoisers live
with the thing they memoise -- ``audio.cache`` owns the CQT, ``transcription.pitch.cache`` owns
the contour -- because that is the layer that knows what "already computed" means and
what it costs. A new cached artifact is a new memoiser next to its producer, not a
change here.

**The key** is a short SHA-256 over the producer's identity and the resolved source
path::

    <root>/<kind>/<source-stem>-<digest><suffix>

Content-addressing by *path* rather than by file contents is deliberate: hashing 100 MB
of audio per lookup would cost more than the lookup saves, and these corpora are
immutable. The tradeoff is that moving a dataset directory invalidates its cache.

The producer's identity must change whenever its output would -- ``htdemucs`` vs
``htdemucs_ft``, a different CQT hop -- or the store serves an artifact built by
something else. See ``AudioTransform.name`` and ``features.SIGNATURE``. That failure is
the quiet kind: a stale CQT is still a valid-looking array of the right shape.

Keying on the resolved audio path is what retired the old ``<split>/<tag>``
partitioning: a stem, a mix and a separated mix already resolve to three different
files, so the condition need not be spelled into the directory tree. One consequence
worth knowing -- every condition shares a single features directory, so a validator
reusing the loader's work is automatic rather than a special case.
"""

import hashlib
from pathlib import Path
from typing import Literal

DEFAULT_ROOT = Path("data/cache")
DIGEST_CHARS = 16

Kind = Literal["features", "contours", "transforms"]
"""The artifact families, as a type rather than a convention.

Three subdirectories, spelled at four call sites. A ``str`` would let a typo open a
fourth silently -- and a store that writes to ``contour/`` while a reader looks in
``contours/`` is a permanent cache miss that costs a pyin pass per track per run and
reports nothing. mypy rejects the typo instead.
"""


def content_key(*parts: object) -> str:
    """A short stable digest over the parts that determine an artifact's content.

    Parts are stringified and NUL-joined, so a separator can never occur inside a part
    and two different tuples cannot collide by concatenation. Truncated to
    ``DIGEST_CHARS``: 64 bits of SHA-256 over a corpus of ~10^4 files is far past where
    a collision is the thing worth worrying about."""
    return hashlib.sha256(
        "\0".join(str(p) for p in parts).encode()
    ).hexdigest()[:DIGEST_CHARS]


class Store:
    """The derived artifacts under one root. Construct once, hand it around."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Store({str(self.root)!r})"

    def dir(self, kind: Kind) -> Path:
        """The directory one family of artifacts lives in."""
        return self.root / kind

    def location(
        self, kind: Kind, producer: str, source: str | Path, suffix: str = ".npz"
    ) -> Path:
        """Where ``producer``'s output for ``source`` lives -- a *pure* function that
        does no work and creates nothing.

        Load-bearing beyond naming a file: the writer and every reader must agree on
        it, and callers need the key *before* deciding whether to compute anything.
        Were it only obtainable by producing the artifact, asking "is this track's CQT
        cached?" would separate the track first -- minutes of Demucs per cache **hit**.

        The source stem stays in the filename so the tree is browsable by eye; the
        digest is what actually disambiguates.
        """
        source = Path(source)
        return self.dir(kind) / f"{source.stem}-{content_key(producer, source.resolve())}{suffix}"
