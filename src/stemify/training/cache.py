"""Disk-cached per-track artifacts: caching *policy* -- keying and get-or-compute.

Split three ways: ``serde.py`` owns the on-disk formats (``TrackExample``, the contour
npz) and ``atomic.py`` the crash-safe write; this owns *when* to recompute and *how*
to key an entry. ``examples.py`` layers per-frame targets on top. Every artifact here
is a pure function of the (transform-resolved) audio, so it is computed once per file
ever and reused across checkpoints, runs, and the frame and sequence datasets alike --
the economy that makes a large train/val set affordable.
"""

import hashlib
import logging
from pathlib import Path

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.features import extract_features
from stemify.serde import TrackExample, load_contour, save_contour
from stemify.transforms import AudioTransform, Identity

logger = logging.getLogger(__name__)


def example_path(cache_dir: Path, name: str, transform_name: str) -> Path:
    """Cache filename for one ``(track, transform)`` TrackExample. Shared by the
    writer (``cache_features``) and any reader (``OnsetValidator``, ``SequenceDataset``)
    so the naming scheme -- including the transform in the key, which is what keeps a
    stem and a demucs pass of the same track from colliding -- lives in one place."""
    return cache_dir / f"{name.replace('/', '_')}__{transform_name}.npz"


def cache_features(
    dataset: SlakhLikeDataset,
    cache_dir: Path,
    transform: AudioTransform | None = None,
) -> list[Path]:
    """Compute and cache each track's CQT as a target-less ``TrackExample``; return the
    cache paths (in dataset order).

    The primitive both training paths build on: the frame path layers per-frame
    targets on top (``prepare_examples``), the sequence path reads the features and
    pairs them with notes. The CQT is the cost, so an existing cache is left untouched
    -- audio is loaded only on a miss. The transform is part of the key, so a stem run
    and a demucs run of the same track never collide, which is what makes source
    domain-randomisation just a matter of pointing at more caches."""
    transform = transform or Identity()
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    paths: list[Path] = []
    computed = 0
    for i, item in enumerate(dataset, start=1):
        path = example_path(cache_dir, item.name, transform.name)
        if not path.exists():
            feats = extract_features(transform.apply(item.audio()))
            TrackExample(item.name, feats.values, {}, feats.frame_times).save(path)
            computed += 1
        paths.append(path)
        # Otherwise silent for hours; a heartbeat so a long separation+CQT pass is
        # visibly progressing. Tagged by the cache dir's name (the source tag), and
        # split computed/cached so a resumed run visibly races through hits.
        if i % 25 == 0 or i == total:
            logger.info(
                "cache_features[%s] %d/%d (%d computed, %d cached)",
                cache_dir.name, i, total, computed, i - computed,
            )
    return paths


def cached_f0(
    baseline: PyinBaseline, audio_path: str | Path, cache_dir: Path | None
) -> Contour:
    """pyin contour for an audio file, memoised to disk.

    The contour is a pure function of the (already transform-resolved) audio and
    pyin's fixed params, so it is computed once per file and reused across every
    checkpoint *and every future run* -- the same disk-cache rationale as the CQT
    features, and the reason a large val set becomes affordable. Keyed on the
    resolved audio path, so each condition's file (clean stem, mix, separated bass)
    caches independently and never collides. ``cache_dir=None`` disables caching.
    """
    if cache_dir is None:
        return baseline.predict_f0(audio_path)
    audio_path = Path(audio_path)
    key = hashlib.sha1(f"{baseline.name}\0{audio_path.resolve()}".encode()).hexdigest()
    dst = cache_dir / f"{audio_path.stem}-{key[:16]}.npz"
    if dst.exists():
        return load_contour(dst)
    contour = baseline.predict_f0(audio_path)
    save_contour(contour, dst)  # atomic_write mkdirs the parent
    return contour
