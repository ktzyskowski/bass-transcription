"""Disk-cached per-track artifacts: the persistence layer, keyed by (track, transform).

Split from ``examples.py`` (which assembles per-frame *targets*) so caching lives in
one place: the on-disk ``TrackExample``, the path keys that keep a stem and a demucs
pass of one track from colliding, the CQT feature cache, and the pyin contour cache.
Every artifact here is a pure function of the (transform-resolved) audio, so it is
computed once per file ever and reused across checkpoints, runs, and the frame and
sequence datasets alike -- the economy that makes a large train/val set affordable.

Assembling targets on top of a cached CQT is ``examples.py``'s job; this module never
imports it, keeping the dependency one-way (targets depend on persistence, not the
reverse).
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.features import extract_features
from stemify.transforms import AudioTransform, Identity

logger = logging.getLogger(__name__)


def features_npy(npz_path: Path) -> Path:
    """Sidecar path holding a TrackExample's features as a bare, uncompressed .npy.

    Features live *outside* the compressed npz so a training window can be sliced
    straight off disk via ``mmap_mode='r'`` -- only the ~100 KB actually read pages
    in, no whole-track decompress, and the OS page cache is shared across DataLoader
    workers rather than each holding its own copy. The npz keeps only the small
    name/frame_times/targets."""
    return npz_path.with_name(npz_path.stem + ".features.npy")


def example_path(cache_dir: Path, name: str, transform_name: str) -> Path:
    """Cache filename for one ``(track, transform)`` TrackExample. Shared by the
    writer (``cache_features``) and any reader (``OnsetValidator``, ``SequenceDataset``)
    so the naming scheme -- including the transform in the key, which is what keeps a
    stem and a demucs pass of the same track from colliding -- lives in one place."""
    return cache_dir / f"{name.replace('/', '_')}__{transform_name}.npz"


@dataclass
class TrackExample:
    """One track's features and per-frame targets, all on ``frame_times``.

    ``features`` may be a read-only memmap (``load(mmap=True)``), so slicing a
    window touches disk lazily instead of holding the whole track in RAM.
    """

    name: str
    features: np.ndarray            # (n_bins, T); a memmap when loaded with mmap=True
    targets: dict[str, np.ndarray]  # each (T,)
    frame_times: np.ndarray         # (T,)

    def save(self, path: Path) -> None:
        np.save(features_npy(path), self.features)  # bare .npy -> memmappable
        np.savez_compressed(
            path, name=self.name, frame_times=self.frame_times,
            **{f"target_{k}": v for k, v in self.targets.items()},
        )

    @classmethod
    def load(cls, path: Path, mmap: bool = False) -> "TrackExample":
        z = np.load(path, allow_pickle=False)
        targets = {
            k[len("target_"):]: z[k] for k in z.files if k.startswith("target_")
        }
        features = np.load(features_npy(path), mmap_mode="r" if mmap else None)
        return cls(str(z["name"]), features, targets, z["frame_times"])


def migrate_example_cache(cache_root: Path) -> int:
    """Rewrite legacy npz (features stored inside) to the split format: features to a
    bare .npy sidecar, the rest kept in a slim npz. No CQT recompute -- just reads
    each npz and re-lays-it-out. Idempotent: already-split files are skipped.
    Returns the number migrated. One-off: droppable once every cache is split."""
    migrated = 0
    for npz in sorted(cache_root.rglob("*.npz")):
        with np.load(npz, allow_pickle=False) as z:
            if "features" not in z.files:  # already split
                continue
            features = z["features"]
            rest = {k: z[k] for k in z.files if k != "features"}
        np.save(features_npy(npz), features)
        np.savez_compressed(npz, **rest)  # overwrite without the features array
        migrated += 1
        if migrated % 200 == 0:
            logger.info("migrated %d examples...", migrated)
    return migrated


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{baseline.name}\0{audio_path.resolve()}".encode()).hexdigest()
    dst = cache_dir / f"{audio_path.stem}-{key[:16]}.npz"
    if dst.exists():
        data = np.load(dst)
        conf = data["confidence"] if "confidence" in data.files else None
        return Contour(times=data["times"], freqs=data["freqs"], confidence=conf)

    contour = baseline.predict_f0(audio_path)
    arrays = {"times": contour.times, "freqs": contour.freqs}
    if contour.confidence is not None:
        arrays["confidence"] = contour.confidence
    # Write-aside-and-rename: an interrupted pyin must not leave a truncated npz
    # that later reads back as a cache hit (same guard as the transforms cache).
    partial = dst.with_name(f"{dst.stem}.{os.getpid()}.partial.npz")
    try:
        np.savez(partial, **arrays)
        partial.replace(dst)
    finally:
        partial.unlink(missing_ok=True)
    return contour
