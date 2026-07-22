"""Slakh tracks -> windowed (CQT features, per-frame targets) for training.

Three stages, deliberately separated so each is testable and the expensive one
is cached:

  - ``build_example`` turns one track's audio + references into a full-track
    (features, targets) pair. Pure given the arrays, so a synthetic contour and a
    few notes exercise the target alignment without touching disk.
  - ``prepare_examples`` runs that over a dataset and caches each result to
    ``.npz`` -- the CQT is the cost, and it must not be recomputed every epoch
    (the same lesson the segmenter sweep taught).
  - ``FrameWindowDataset`` serves fixed-length windows over the cached examples;
    the GRU needs a few seconds of context and the batch needs a fixed length.

Targets live on the CQT frame grid (``frame_times``), built by ``labels.py``;
features and every target therefore share one grid, and none of them assume it
lines up with the 10 ms reference grid.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.features import HOP_LENGTH, SAMPLE_RATE, cqt, load_audio
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
    Returns the number migrated."""
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


def frame_times_for(features: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Centre time of each CQT frame -- the grid every target is built on."""
    return librosa.frames_to_time(
        np.arange(features.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )


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
    computed = 0  # cache misses actually computed (the slow ones; the rest are hits)
    for i, item in enumerate(dataset, start=1):
        path = example_path(cache_dir, item.name, transform.name)
        if not path.exists():
            audio = load_audio(transform.apply(item.audio()))
            features = cqt(audio)
            example = build_example(
                item.name, features, frame_times_for(features),
                item.reference(), item.reference_notes(), heads,
                onset_tolerance_frames,
            )
            example.save(path)
            computed += 1
        paths.append(path)
        # Prep is otherwise silent for hours; print a heartbeat so a long
        # separation+CQT pass is visibly progressing. Tagged by the cache dir's
        # name (the source tag: stem/mix/demucs -- transform.name can't tell stem
        # from mix, both Identity), and split computed vs cached so a resumed run
        # visibly races through hits and slows on real work.
        if i % 25 == 0 or i == total:
            logger.info(
                "prepare[%s] %d/%d (%d computed, %d cached)",
                cache_dir.name, i, total, computed, i - computed,
            )
    return paths


class FrameWindowDataset(Dataset):
    """Fixed-length windows over cached ``TrackExample``s.

    Windows are enumerated up front (deterministic), so training order is a
    shuffle of a stable set rather than a moving target. A track shorter than one
    window yields a single zero-padded window rather than being dropped.
    """

    def __init__(
        self,
        example_paths: Sequence[Path],
        heads: Iterable[str],
        window_frames: int = 256,
        hop_frames: int = 128,
    ) -> None:
        self.paths = list(example_paths)
        self.heads = list(heads)
        self.window = window_frames
        self._index: list[tuple[int, int]] = []
        for ei, p in enumerate(self.paths):
            n_frames = int(np.load(p, allow_pickle=False)["frame_times"].shape[0])
            if n_frames <= window_frames:
                self._index.append((ei, 0))
            else:
                for start in range(0, n_frames - window_frames + 1, hop_frames):
                    self._index.append((ei, start))
        self._cache: dict[int, TrackExample] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _example(self, ei: int) -> TrackExample:
        # mmap: the cached TrackExample holds a memmap (no resident feature data) plus
        # the small targets/frame_times, so this cache stays cheap even unbounded --
        # the big features are paged in per-window by the OS, shared across workers.
        if ei not in self._cache:
            self._cache[ei] = TrackExample.load(self.paths[ei], mmap=True)
        return self._cache[ei]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ei, start = self._index[i]
        ex = self._example(ei)
        # np.asarray copies just this window out of the memmap into a real array.
        feat = np.asarray(ex.features[:, start:start + self.window])
        x = np.zeros((ex.features.shape[0], self.window), dtype=np.float32)
        x[:, : feat.shape[1]] = feat
        y: dict[str, torch.Tensor] = {}
        for h in self.heads:
            t = ex.targets[h][start:start + self.window]
            padded = np.zeros(self.window, dtype=np.float32)
            padded[: t.shape[0]] = t
            y[h] = torch.from_numpy(padded)
        return torch.from_numpy(x).unsqueeze(0), y  # (1, n_bins, W), {head: (W,)}


def collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack windows into (B, 1, n_bins, W) and each target into (B, W)."""
    xs = torch.stack([x for x, _ in batch])
    heads = batch[0][1].keys()
    ys = {h: torch.stack([y[h] for _, y in batch]) for h in heads}
    return xs, ys