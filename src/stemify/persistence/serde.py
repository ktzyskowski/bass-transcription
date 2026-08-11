"""Object <-> disk formats for the cached artifacts. Serialization, not policy.

``cache.py`` owns *when* to recompute and *how* to key an entry; this owns *what
bytes* an artifact becomes. Splitting them keeps two things clean: a neutral value
type (``Contour``) gains no disk knowledge -- its format lives here as a free
function, not a method that would couple the value to npz -- and a new cached
artifact is a serde here, not a change to the cache.

Every write goes through ``atomic_write``, so an interrupted save never leaves a
truncated file that the next run trusts as a cache hit.

``CachedItem`` lives here rather than beside the cache because it *is* a persistence
container -- its whole reason to exist is the on-disk split-layout below -- so it
belongs with its format, unlike ``Contour``/``NoteEvent``/``Frames`` which are values
that happen to be persisted.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from stemify.platform.atomic import atomic_write


def _write_npy(dest: Path, arr: np.ndarray) -> None:
    # Through an open handle: np.save appends '.npy' to a *filename* that lacks it,
    # which would miss the atomic temp path (``...npy.<pid>.partial``) entirely.
    with open(dest, "wb") as f:
        np.save(f, arr)


def write_npz(dest: Path, *, compressed: bool, **arrays: npt.ArrayLike) -> None:
    """Write arrays as an npz. Public because callers that rewrite *part* of a cached
    example (the targets backfill) need the same handle trick and the same atomicity,
    and reimplementing either is how a truncated file ends up trusted as a cache hit.

    Same reason as _write_npy: a handle stops np.savez appending '.npz'.

    ``ArrayLike`` rather than ``ndarray``: ``CachedItem.save`` passes its ``name``
    through here as a ``str``, which ``np.savez`` stores as a 0-d array and ``load``
    reads back with ``str(z["name"])``. That has always worked; the annotation simply
    did not describe it."""
    with open(dest, "wb") as f:
        # The ignore is a stub artifact, not a defect: numpy spells these
        # ``savez(file, *args, allow_pickle=..., **kwds)``, and a ``**mapping`` splat is
        # checked against *every* keyword the callee declares -- so the array values are
        # tested against ``allow_pickle: bool`` even though they can only land in
        # ``**kwds``. There is no call shape that avoids it.
        write = np.savez_compressed if compressed else np.savez
        write(f, **arrays)  # type: ignore[arg-type]


def features_npy(npz_path: Path) -> Path:
    """Sidecar path holding a CachedItem's features as a bare, uncompressed .npy.

    Features live *outside* the compressed npz so a training window can be sliced
    straight off disk via ``mmap_mode='r'`` -- only the ~100 KB actually read pages
    in, no whole-track decompress, and the OS page cache is shared across DataLoader
    workers rather than each holding its own copy. The npz keeps only the small
    name/frame_times/targets."""
    return npz_path.with_name(npz_path.stem + ".features.npy")


@dataclass
class CachedItem:
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
            lambda p: write_npz(
                p, compressed=True, name=self.name, frame_times=self.frame_times,
                **{f"target_{k}": v for k, v in self.targets.items()},
            ),
        )

    @classmethod
    def load(cls, path: Path, mmap: bool = False) -> "CachedItem":
        z = np.load(path, allow_pickle=False)
        targets = {
            k[len("target_"):]: z[k] for k in z.files if k.startswith("target_")
        }
        features = np.load(features_npy(path), mmap_mode="r" if mmap else None)
        return cls(str(z["name"]), features, targets, z["frame_times"])

