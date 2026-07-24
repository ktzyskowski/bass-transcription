"""Assemble per-frame training targets on top of a cached CQT.

The target layer, on top of the persistence layer (``cache.py``) and below the
windowing that serves batches (``datasets/frame.py``). Two pieces, split so each is
testable and the expensive one (the CQT) is cached once:

  - ``build_example`` turns one track's features + references into a full-track
    (features, targets) ``TrackExample``. Pure given the arrays, so a synthetic
    contour and a few notes exercise the target alignment without touching disk.
  - ``prepare_examples`` caches each track's CQT (via ``cache_features``) and layers
    the requested heads' targets on top. Self-healing: a cached example missing a
    newly-requested head's target gains it from the reference alone, no CQT recompute.

Targets live on the CQT frame grid (``frame_times``); features and every target
share that one grid, none assume the 10 ms reference grid.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.data.base import EvalItem
from stemify.notes import NoteEvent
from stemify.serde import TrackExample
from stemify.training.cache import cache_features
from stemify.training.labels import onset_targets, pitch_targets, voicing_targets
from stemify.transforms import AudioTransform


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


def prepare_examples(
    dataset: SlakhLikeDataset,
    heads: Iterable[str],
    cache_dir: Path,
    transform: AudioTransform | None = None,
    onset_tolerance_frames: int = 1,
) -> list[Path]:
    """Cache each track's CQT and layer the requested heads' targets on top; return
    the cache paths.

    The CQT is cached once by ``cache_features`` (audio touched only on a miss); this
    then backfills each requested head's target from the reference alone. Self-healing:
    a track already cached with fewer heads gains only the missing targets, no CQT
    recompute -- so widening the head set (adding voicing to an onset-only cache) reuses
    every CQT and pays only a MIDI parse per track.
    """
    heads = list(heads)
    paths = cache_features(dataset, cache_dir, transform)
    for item, path in zip(dataset, paths, strict=True):
        _ensure_targets(path, item, heads, onset_tolerance_frames)
    return paths


def _ensure_targets(
    path: Path, item: EvalItem, heads: Iterable[str], onset_tolerance_frames: int
) -> bool:
    """Add any requested head-targets a cached example is missing -- without
    recomputing its CQT. When the head set grows (e.g. adding voicing to an
    onset-only cache) the features are unchanged, only the labels are new, so this
    recomputes just the missing targets from the reference on the cached frame grid
    and rewrites the slim npz; the .npy feature sidecar is left untouched. So a
    voicing run reuses the whole existing CQT cache and only pays a MIDI parse per
    track. Returns whether anything was added."""
    with np.load(path, allow_pickle=False) as z:
        have = {k[len("target_") :] for k in z.files if k.startswith("target_")}
        missing = [h for h in heads if h not in have]
        if not missing:
            return False
        arrays = {k: z[k] for k in z.files}
        frame_times = z["frame_times"]
    added = build_example(
        item.name,
        np.empty((1, frame_times.size), dtype=np.float32),
        frame_times,
        item.reference(),
        item.reference_notes(),
        missing,
        onset_tolerance_frames,
    ).targets
    for head, target in added.items():
        arrays[f"target_{head}"] = target
    np.savez_compressed(path, **arrays)
    return True
