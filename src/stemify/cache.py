"""Every derived artifact for a corpus, under one root: layout, keying, get-or-compute.

Three things get cached on the way from audio to a trained model, and they are all
pure functions of some *resolved audio file* plus the identity of whatever produced
them: separated audio, a pyin contour, a CQT. ``Cache`` is the one object that knows
where each lives and how each is named.

**Why an object rather than free functions taking paths.** The previous shape threaded
``cache_dir``, ``contour_cache`` and ``feature_cache`` separately through seven
signatures, and the *layout* -- which subdirectory holds what -- was spelled out in
``training/train.py``, which is why the async validator needed a paragraph explaining
which cache it was reusing from the BCE loader. One object collapses those parameters
and moves layout here, where it can be stated once.

**Why the root package and not ``training/``.** The evaluation harness runs the same
transforms over the same corpora and currently recomputes pyin and the CQT on every
launch, having no way to reach what training already built. Caching is not a training
concern that eval borrows; it is a project-wide one.

**The keying convention** (``content_key``) is a short SHA-256 over the producer's
identity and the resolved source path, giving::

    <dir>/<source-stem>-<digest><suffix>

Content-addressing by *path* rather than file contents is deliberate: hashing 100 MB
of audio per lookup would cost more than the lookup saves, and these corpora are
immutable. The tradeoff is that moving a dataset directory invalidates its cache.

The producer's identity must change whenever its output would -- ``htdemucs`` vs
``htdemucs_ft``, a different CQT hop -- or the cache silently serves an artifact built
by something else. See ``AudioTransform.name`` and ``features.SIGNATURE``.

Keying features on the resolved audio path is what retired the old
``cache_dir/<split>/<tag>`` partitioning: a stem, a mix and a separated mix already
resolve to three different files, so the condition no longer needs to be spelled into
the directory tree. One consequence worth knowing -- every condition now shares a
single features directory, so the validator reusing the loader's work is automatic
rather than a special case for the stem tag.
"""

import hashlib
import logging
from pathlib import Path

from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.features import SIGNATURE as CQT_SIGNATURE
from stemify.features import extract_features
from stemify.pitch import PitchEstimator
from stemify.serde import CachedItem, load_contour, save_contour
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/cache")
DIGEST_CHARS = 16


def content_key(*parts: object) -> str:
    """A short stable digest over the parts that determine an artifact's content.

    Parts are stringified and NUL-joined, so a separator can never occur inside a part
    and two different tuples cannot collide by concatenation. Truncated to
    ``DIGEST_CHARS``: 64 bits of SHA-256 over a corpus of ~10^4 files is far past where
    a collision is the thing worth worrying about."""
    return hashlib.sha256(
        "\0".join(str(p) for p in parts).encode()
    ).hexdigest()[:DIGEST_CHARS]


class Cache:
    """The derived artifacts under one root. Construct once, hand it around."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Cache({str(self.root)!r})"

    # -- layout -------------------------------------------------------------------
    # Stated here rather than assembled by callers, so "where do the contours live"
    # has one answer and a reader of train.py is not reverse-engineering it from
    # path arithmetic.

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def contours_dir(self) -> Path:
        return self.root / "contours"

    @property
    def transforms_dir(self) -> Path:
        return self.root / "transforms"

    def separator(
        self, model: str = "htdemucs", stem: str = "bass", device: str | None = None
    ) -> DemucsSeparator:
        """A ``DemucsSeparator`` writing into this cache.

        The cache owns the *directory*; the transform still does its own keying and
        writing. That keeps the dependency one-way -- this module imports
        ``transforms``, never the reverse -- so the light audio path never pulls the
        cache in, while every byte under ``root`` is still accounted for here."""
        return DemucsSeparator(
            model=model, stem=stem, device=device, cache_dir=self.transforms_dir
        )

    # -- features -----------------------------------------------------------------

    def feature_path(self, audio: str | Path) -> Path:
        """Where the CQT for an already transform-resolved audio file is cached.

        Pure, and the writer and every reader must agree on it, which is why it is a
        method rather than each caller rebuilding the name."""
        audio = Path(audio)
        return self.features_dir / f"{audio.stem}-{content_key(CQT_SIGNATURE, audio.resolve())}.npz"

    def ensure_features(
        self,
        dataset: SlakhLikeDataset,
        transform: AudioTransform | None = None,
        role: str = "bass",
    ) -> list[Path]:
        """Compute and cache each track's CQT as a target-less ``CachedItem``;
        return the cache paths in dataset order.

        The primitive both training paths build on: the frame path layers per-frame
        targets on top (``prepare_examples``), the sequence path reads the features
        and pairs them with notes. The CQT is the cost, so an existing cache is left
        untouched -- audio is loaded only on a miss.

        ``role`` picks *which* of the track's audio files to featurise (see
        ``Sources``). It defaults to the one the item is about, so single-input
        training is unaffected; a second call with ``role="mix"`` builds the extra
        channel a mix-conditioned encoder reads. Nothing here needs to know the two
        belong together -- each is keyed on its own resolved path, so they cannot
        collide and either can already be a hit from another condition's run.

        The key comes from ``transform.cache_path``, which resolves where the audio
        *would* be without producing it. Using ``apply`` here would separate every
        track through Demucs just to discover its CQT was already cached."""
        transform = transform or Identity()
        self.features_dir.mkdir(parents=True, exist_ok=True)
        total, computed = len(dataset), 0
        paths: list[Path] = []
        for i, item in enumerate(dataset, start=1):
            source = item.sources().role(role)
            path = self.feature_path(transform.cache_path(source))
            if not path.exists():
                feats = extract_features(transform.apply(source))
                CachedItem(item.name, feats.values, {}, feats.frame_times).save(path)
                computed += 1
            paths.append(path)
            # Otherwise silent for hours; a heartbeat so a long separation+CQT pass is
            # visibly progressing. Tagged by role and transform (the channel being
            # built) and split computed/cached so a resumed run visibly races through
            # hits -- which is what a second channel over an already-warm cache looks
            # like, and worth being able to see rather than infer.
            if i % 25 == 0 or i == total:
                logger.info(
                    "features[%s/%s] %d/%d (%d computed, %d cached)",
                    role, transform.name, i, total, computed, i - computed,
                )
        return paths

    # -- contours -----------------------------------------------------------------

    def contour(self, estimator: PitchEstimator, audio: str | Path) -> Contour:
        """A ``PitchEstimator``'s contour for an audio file, memoised to disk.

        Pure in the (already transform-resolved) audio and the estimator's fixed
        params, so it is computed once per file ever and reused across every
        checkpoint *and every future run* -- the same rationale as the CQT, and what
        makes a large val set affordable. Each condition's file caches independently
        and never collides, since the resolved path is in the key."""
        audio = Path(audio)
        dst = self.contours_dir / f"{audio.stem}-{content_key(estimator.name, audio.resolve())}.npz"
        if dst.exists():
            return load_contour(dst)
        contour = estimator.estimate(audio)
        save_contour(contour, dst)  # atomic_write mkdirs the parent
        return contour
