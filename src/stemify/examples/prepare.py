"""Layering labels onto a cached CQT, and keeping them in step as heads are added.

The expensive half (the CQT) is `audio.cache`'s and the pure half (the target arrays)
is `music.labels`'. What is left here is the part that is neither: walking a corpus,
pairing each item with its cache entry, and back-filling the targets a run asks for
that the entry does not yet carry.

That back-fill is why this exists at all. When a head set grows -- adding voicing to an
onset-only cache -- the features are unchanged and only the labels are new, so a run
reuses every CQT and pays a MIDI parse per track instead of a full re-featurisation of
the corpus.

Targets live on the CQT frame grid (``frame_times``); features and every target share
that one grid, and none assume the 10 ms reference grid the metrics use.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from stemify.audio.cache import ensure_features
from stemify.audio.transforms import AudioTransform, Identity
from stemify.corpus.base import Dataset as SlakhLikeDataset, NoteAnnotated
from stemify.music.labels import build_targets
from stemify.persistence.serde import write_npz
from stemify.persistence.store import Store
from stemify.platform.atomic import atomic_write


def prepare_examples(
    dataset: SlakhLikeDataset[NoteAnnotated],
    heads: Iterable[str],
    store: Store,
    transform: AudioTransform | None = None,
    onset_tolerance_frames: int = 1,
) -> list[Path]:
    """Store each track's CQT and layer the requested heads' targets on top; return
    the store paths.

    The CQT is cached once by ``audio.cache.ensure_features`` (audio touched only on a
    miss); this then backfills each requested head's target from the reference alone.
    Self-healing: a track already cached with fewer heads gains only the missing
    targets, no CQT recompute -- so widening the head set (adding voicing to an
    onset-only store) reuses every CQT and pays only a MIDI parse per track.
    """
    items = list(dataset)
    paths = ensure_features(
        store, transform or Identity(), [i.sources().bass for i in items], label="bass"
    )
    ensure_targets(items, paths, heads, onset_tolerance_frames)
    return paths


def ensure_targets(
    items: Iterable[NoteAnnotated],
    paths: Sequence[Path],
    heads: Iterable[str],
    onset_tolerance_frames: int = 1,
) -> None:
    """Back-fill each requested head's target onto already-cached examples, in place.

    Split out of ``prepare_examples`` because the structured path reaches the same
    store by a different route -- it resolves channels through
    ``Condition.channel_paths``, so it holds the paths already and needs only the
    labels layered on. Both callers must share the ownership guard below, which is
    exactly the kind of thing that drifts once it is copied.
    """
    heads = list(heads)
    if not heads:
        return
    # Features are content-addressed on the resolved audio, so two items sharing one
    # audio file share one store entry. That is right for the CQT (identical audio,
    # identical features) but wrong for *targets*: each item has its own reference
    # notes, and only the first would ever be written -- the second's `_ensure_targets`
    # finds the head already present and skips, silently mislabelling the example.
    #
    # It cannot happen today (mix-based sources are filtered to single-bass tracks, so
    # every item has a distinct audio file), which is exactly why it needs a guard
    # rather than a comment: relaxing that filter would corrupt labels with no error.
    owner: dict[Path, str] = {}
    for item, path in zip(items, paths, strict=True):
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
    onset-only store) the features are unchanged, only the labels are new, so this
    recomputes just the missing targets from the reference on the cached frame grid
    and rewrites the slim npz; the .npy feature sidecar is left untouched. So a
    voicing run reuses the whole existing CQT store and only pays a MIDI parse per
    track. Returns whether anything was added."""
    with np.load(path, allow_pickle=False) as z:
        have = {k[len("target_") :] for k in z.files if k.startswith("target_")}
        missing = [h for h in heads if h not in have]
        if not missing:
            return False
        arrays = {k: z[k] for k in z.files}
        frame_times = z["frame_times"]
    added = build_targets(
        frame_times,
        item.reference_contour(),
        item.reference_notes(),
        missing,
        onset_tolerance_frames,
    )
    for head, target in added.items():
        arrays[f"target_{head}"] = target
    # Atomic like every other store write: this rewrites an npz a later run keys on,
    # so an interrupted backfill must not leave a truncated file that then reads as a
    # store hit with some targets silently missing.
    atomic_write(path, lambda p: write_npz(p, compressed=True, **arrays))
    return True
