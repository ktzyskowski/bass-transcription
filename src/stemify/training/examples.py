"""Slakh tracks -> cached per-track (CQT features, per-frame targets).

The preparation/cache layer, separate from the windowing that serves batches
(``dataset.py``). Three pieces, deliberately split so each is testable and the
expensive one is cached:

  - ``build_example`` turns one track's features + references into a full-track
    (features, targets) pair. Pure given the arrays, so a synthetic contour and a
    few notes exercise the target alignment without touching disk.
  - ``TrackExample`` is that pair on disk: features in a bare, memmappable .npy
    sidecar, the small targets/frame_times in a slim compressed npz beside it.
  - ``prepare_examples`` runs the build over a dataset and caches each result --
    the CQT is the cost, and it must not be recomputed every epoch. It is
    self-healing: a cached example missing a newly-requested head's target gains
    it from the reference alone, no CQT recompute.

Targets live on the CQT frame grid (``frame_times``); features and every target
share that one grid, none assume the 10 ms reference grid.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.features import extract_features
from stemify.notes import NoteEvent
from stemify.training.labels import onset_targets, pitch_targets, voicing_targets
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


def build_example(
    name: str,
    features: np.ndarray,
    frame_times: np.ndarray,
    reference: Contour,
    notes: Sequence[NoteEvent],
    heads: Iterable[str],
    onset_tolerance_frames: int = 1,
) -> TrackExample:
    """Assemble the targets each requested head needs, on ``frame_times``.

    ``reference`` is the *gated* reference contour (voicing + pitch); ``notes``
    give the exact onsets. Labels always come from the clean stem's truth, no
    matter which audio the features came from.
    """
    heads = set(heads)
    targets: dict[str, np.ndarray] = {}
    if "onset" in heads:
        onsets = np.array([n.onset for n in notes], dtype=float)
        targets["onset"] = onset_targets(onsets, frame_times, onset_tolerance_frames)
    if "voicing" in heads:
        targets["voicing"] = voicing_targets(
            reference.times, reference.voiced, frame_times
        )
    if "pitch" in heads:
        midi = np.full(reference.freqs.shape, np.nan, dtype=float)
        voiced = reference.freqs > 0
        midi[voiced] = librosa.hz_to_midi(reference.freqs[voiced])
        targets["pitch"] = pitch_targets(reference.times, midi, frame_times)
    return TrackExample(name, features, targets, frame_times)


def example_path(cache_dir: Path, name: str, transform_name: str) -> Path:
    """Cache filename for one ``(track, transform)`` TrackExample. Shared by the
    writer (``prepare_examples``) and any reader (``OnsetValidator``) so the naming
    scheme -- including the transform in the key, which is what keeps a stem and a
    demucs pass of the same track from colliding -- lives in exactly one place."""
    return cache_dir / f"{name.replace('/', '_')}__{transform_name}.npz"


def prepare_examples(
    dataset: SlakhLikeDataset,
    heads: Iterable[str],
    cache_dir: Path,
    transform: AudioTransform | None = None,
    onset_tolerance_frames: int = 1,
) -> list[Path]:
    """Compute and cache a ``TrackExample`` per item; return the cache paths.

    Cache key includes the transform, so a stem run and a demucs run of the same
    track do not collide -- that is what makes source domain-randomisation just a
    matter of pointing the dataset at more caches.
    """
    transform = transform or Identity()
    cache_dir.mkdir(parents=True, exist_ok=True)
    heads = list(heads)
    total = len(dataset)
    paths: list[Path] = []
    computed = 0     # cache misses: full CQT recompute (the slow ones)
    backfilled = 0   # cache hits missing a target: cheap label-only rewrite, no CQT
    for i, item in enumerate(dataset, start=1):
        path = example_path(cache_dir, item.name, transform.name)
        if not path.exists():
            feats = extract_features(transform.apply(item.audio()))
            example = build_example(
                item.name, feats.values, feats.frame_times,
                item.reference(), item.reference_notes(), heads,
                onset_tolerance_frames,
            )
            example.save(path)
            computed += 1
        elif _ensure_targets(path, item, heads, onset_tolerance_frames):
            backfilled += 1
        paths.append(path)
        # Prep is otherwise silent for hours; print a heartbeat so a long
        # separation+CQT pass is visibly progressing. Tagged by the cache dir's
        # name (the source tag: stem/mix/demucs -- transform.name can't tell stem
        # from mix, both Identity), and split computed/backfilled/cached so a
        # resumed run visibly races through hits and slows on real work.
        if i % 25 == 0 or i == total:
            logger.info(
                "prepare[%s] %d/%d (%d computed, %d backfilled, %d cached)",
                cache_dir.name, i, total, computed, backfilled,
                i - computed - backfilled,
            )
    return paths


def _ensure_targets(
    path: Path, item: object, heads: Iterable[str], onset_tolerance_frames: int
) -> bool:
    """Add any requested head-targets a cached example is missing -- without
    recomputing its CQT. When the head set grows (e.g. adding voicing to an
    onset-only cache) the features are unchanged, only the labels are new, so this
    recomputes just the missing targets from the reference on the cached frame grid
    and rewrites the slim npz; the .npy feature sidecar is left untouched. So a
    voicing run reuses the whole existing CQT cache and only pays a MIDI parse per
    track. Returns whether anything was added."""
    with np.load(path, allow_pickle=False) as z:
        have = {k[len("target_"):] for k in z.files if k.startswith("target_")}
        missing = [h for h in heads if h not in have]
        if not missing:
            return False
        arrays = {k: z[k] for k in z.files}
        frame_times = z["frame_times"]
    added = build_example(
        item.name, np.empty((1, frame_times.size), dtype=np.float32), frame_times,
        item.reference(), item.reference_notes(), missing, onset_tolerance_frames,
    ).targets
    for head, target in added.items():
        arrays[f"target_{head}"] = target
    np.savez_compressed(path, **arrays)
    return True
