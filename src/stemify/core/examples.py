"""Assemble per-frame training targets on top of a cached CQT.

The target layer, on top of the persistence layer (``cache.py``) and below the
windowing that serves batches (``frame/dataset.py``). Two pieces, split so each is
testable and the expensive one (the CQT) is cached once:

  - ``build_example`` turns one track's features + references into a full-track
    (features, targets) ``CachedItem``. Pure given the arrays, so a synthetic
    contour and a few notes exercise the target alignment without touching disk.
  - ``prepare_examples`` caches each track's CQT (via ``Cache.ensure_features``) and layers
    the requested heads' targets on top. Self-healing: a cached example missing a
    newly-requested head's target gains it from the reference alone, no CQT recompute.

Targets live on the CQT frame grid (``frame_times``); features and every target
share that one grid, none assume the 10 ms reference grid.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import librosa
import numpy as np

from stemify.platform.atomic import atomic_write
from stemify.core.cache import Cache
from stemify.music.contour import Contour
from stemify.corpus.base import Dataset as SlakhLikeDataset, NoteAnnotated
from stemify.music.notes import NoteEvent
from stemify.core.serde import CachedItem, write_npz
from stemify.music.labels import onset_targets, pitch_targets, voicing_targets
from stemify.audio.transforms import AudioTransform


def build_example(
    name: str,
    features: np.ndarray,
    frame_times: np.ndarray,
    reference: Contour,
    notes: Sequence[NoteEvent],
    heads: Iterable[str],
    onset_tolerance_frames: int = 1,
) -> CachedItem:
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
    return CachedItem(name, features, targets, frame_times)


def prepare_examples(
    dataset: SlakhLikeDataset[NoteAnnotated],
    heads: Iterable[str],
    cache: Cache,
    transform: AudioTransform | None = None,
    onset_tolerance_frames: int = 1,
) -> list[Path]:
    """Cache each track's CQT and layer the requested heads' targets on top; return
    the cache paths.

    The CQT is cached once by ``Cache.ensure_features`` (audio touched only on a miss);
    this then backfills each requested head's target from the reference alone.
    Self-healing: a track already cached with fewer heads gains only the missing
    targets, no CQT recompute -- so widening the head set (adding voicing to an
    onset-only cache) reuses every CQT and pays only a MIDI parse per track.
    """
    paths = cache.ensure_features(dataset, transform)
    ensure_targets(dataset, paths, heads, onset_tolerance_frames)
    return paths


def ensure_targets(
    dataset: SlakhLikeDataset[NoteAnnotated],
    paths: Sequence[Path],
    heads: Iterable[str],
    onset_tolerance_frames: int = 1,
) -> None:
    """Back-fill each requested head's target onto already-cached examples, in place.

    Split out of ``prepare_examples`` because the structured path reaches the same
    cache by a different route -- it resolves channels through
    ``Condition.channel_paths``, so it holds the paths already and needs only the
    labels layered on. Both callers must share the ownership guard below, which is
    exactly the kind of thing that drifts once it is copied.
    """
    heads = list(heads)
    if not heads:
        return
    # Features are content-addressed on the resolved audio, so two items sharing one
    # audio file share one cache entry. That is right for the CQT (identical audio,
    # identical features) but wrong for *targets*: each item has its own reference
    # notes, and only the first would ever be written -- the second's `_ensure_targets`
    # finds the head already present and skips, silently mislabelling the example.
    #
    # It cannot happen today (mix-based sources are filtered to single-bass tracks, so
    # every item has a distinct audio file), which is exactly why it needs a guard
    # rather than a comment: relaxing that filter would corrupt labels with no error.
    owner: dict[Path, str] = {}
    for item, path in zip(dataset, paths, strict=True):
        if owner.setdefault(path, item.name) != item.name:
            raise ValueError(
                f"{item.name!r} and {owner[path]!r} share cached features at {path} "
                "but carry different targets; their audio must differ, or the targets "
                "must be keyed separately from the features."
            )
        _ensure_targets(path, item, heads, onset_tolerance_frames)


def _ensure_targets(
    path: Path, item: NoteAnnotated, heads: Iterable[str], onset_tolerance_frames: int
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
    # Atomic like every other cache write: this rewrites an npz a later run keys on,
    # so an interrupted backfill must not leave a truncated file that then reads as a
    # cache hit with some targets silently missing.
    atomic_write(path, lambda p: write_npz(p, compressed=True, **arrays))
    return True
