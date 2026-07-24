"""Object <-> disk formats for the cached artifacts. Serialization, not policy.

``cache.py`` owns *when* to recompute and *how* to key an entry; this owns *what
bytes* an artifact becomes. Splitting them keeps two things clean: a neutral value
type (``Contour``) gains no disk knowledge -- its format lives here as a free
function, not a method that would couple the value to npz -- and a new cached
artifact is a serde here, not a change to the cache.

Every write goes through ``atomic_write``, so an interrupted save never leaves a
truncated file that the next run trusts as a cache hit.

``TrackExample`` lives here rather than beside the cache because it *is* a persistence
container -- its whole reason to exist is the on-disk split-layout below -- so it
belongs with its format, unlike ``Contour``/``NoteEvent``/``Cues`` which are values
that happen to be persisted.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.atomic import atomic_write
from stemify.contour import Contour


def _write_npy(dest: Path, arr: np.ndarray) -> None:
    # Through an open handle: np.save appends '.npy' to a *filename* that lacks it,
    # which would miss the atomic temp path (``...npy.<pid>.partial``) entirely.
    with open(dest, "wb") as f:
        np.save(f, arr)


def _write_npz(dest: Path, *, compressed: bool, **arrays: np.ndarray) -> None:
    # Same reason as _write_npy: a handle stops np.savez appending '.npz'.
    with open(dest, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)


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
        # Features (the .npy sidecar) first, metadata (the .npz) last: a reader keys
        # on the npz existing, so a crash between the two atomic renames leaves no
        # npz -> a clean miss that recomputes, never a stale-features cache hit.
        atomic_write(features_npy(path), lambda p: _write_npy(p, self.features))
        atomic_write(
            path,
            lambda p: _write_npz(
                p, compressed=True, name=self.name, frame_times=self.frame_times,
                **{f"target_{k}": v for k, v in self.targets.items()},
            ),
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
    return migrated


def save_contour(contour: Contour, path: Path) -> None:
    """Persist a pyin ``Contour`` to npz (times/freqs, and confidence when present)."""
    arrays = {"times": contour.times, "freqs": contour.freqs}
    if contour.confidence is not None:
        arrays["confidence"] = contour.confidence
    atomic_write(path, lambda p: _write_npz(p, compressed=False, **arrays))


def load_contour(path: Path) -> Contour:
    """Read back a ``Contour`` saved by ``save_contour``."""
    data = np.load(path)
    conf = data["confidence"] if "confidence" in data.files else None
    return Contour(times=data["times"], freqs=data["freqs"], confidence=conf)
