"""Cached note-level validation for the async validator.

The async note-level eval is dominated by pyin, but pyin's pitch contour is fixed
across checkpoints -- only the onset model changes. ``OnsetValidator`` runs pyin
(and the CQT) over the val tracks *once* at construction and holds the results,
so each subsequent score is just a model forward, a peak-pick and the grouping --
seconds, not minutes. That is what makes async validation every few epochs cheap
enough to leave on.

``score_examples`` is the pure core (given precomputed examples), so the scoring
is tested with synthetic inputs and a fake model, no audio.
"""

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset
from stemify.evaluation.metrics import PRF, Fragmentation
from stemify.features import Features, extract_features
from stemify.notes import NoteEvent
from stemify.training.examples import (
    example_path,
    features_npy,
    prepare_examples,
)
from stemify.transforms import AudioTransform, Identity

# A segmentation step: cached features + their contour -> notes. The validator
# depends on this shape, not on the concrete LearnedSegmenter/FrameTagger -- the
# caller wraps a checkpoint and passes its bound ``segment_features``.
SegmentFn = Callable[[Contour, Features], list[NoteEvent]]


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


@dataclass
class ValExample:
    """One val track, everything fixed across checkpoints precomputed.

    Features are held either in memory (``features``, e.g. tests or the no-cache
    path) or lazily on disk (``feature_path``, the cached path). Holding all val
    features in memory for a full-scale val set is ~21 GB across three conditions,
    so when they are cached to disk we keep only the path and load the tensor
    transiently per score -- see ``load_features``. Only ``frame_times`` (small) is
    kept resident, since scoring needs it for onset timing."""

    frame_times: np.ndarray       # (T,)
    contour: Contour              # pyin pitch, fixed
    notes: list[NoteEvent]        # reference notes
    features: np.ndarray | None = None      # (n_bins, T) CQT, if resident
    feature_path: Path | None = None        # else load from disk on demand

    def load_features(self) -> Features:
        """The track's CQT paired with its frame grid, from memory or disk. The disk
        read is transient -- freed after each score -- so memory stays flat over the
        run. ``feature_path`` is the TrackExample npz; the CQT is its .npy sidecar."""
        values = self.features
        if values is None:
            values = np.load(features_npy(self.feature_path))
        return Features(values, self.frame_times)


def score_examples(
    segment: SegmentFn,
    examples: Sequence[ValExample],
) -> dict[str, float]:
    """Score a segmentation function over precomputed val examples: for each, hand
    its contour + cached CQT to ``segment`` and compare the notes it returns to the
    references.

    The validator owns no segmentation logic -- it just calls the function it was
    given (typically a ``LearnedSegmenter.segment_features``, the same class
    deployment uses), so how notes are made lives entirely in the segmenter and eval
    can't drift from deploy. Pure given ``examples`` and ``segment``."""
    refs: list[list[NoteEvent]] = []
    ests: list[list[NoteEvent]] = []
    for ex in examples:
        ests.append(segment(ex.contour, ex.load_features()))
        refs.append(ex.notes)
    prf = PRF().calculate(refs, ests).scalars
    frag = Fragmentation().calculate(refs, ests).scalars
    return {
        "note_f1": prf["f1"],
        "note_precision": prf["precision"],
        "note_recall": prf["recall"],
        "fragmentation": float(frag["fragmentation"]),
    }


class OnsetValidator:
    """Precompute pyin contours + CQT + reference notes once; score models cheaply."""

    def __init__(
        self,
        dataset: Dataset,
        transform: AudioTransform | None = None,
        contour_cache: Path | None = None,
        feature_cache: Path | None = None,
    ) -> None:
        transform = transform or Identity()
        baseline = PyinBaseline()

        # The CQT is the same the training pipeline already caches. With
        # ``feature_cache`` set, populate it once (``prepare_examples``, disk-cached
        # per (track, transform)) and read features back rather than recomputing in
        # memory every launch -- and reuse the stem cache the BCE loader already
        # built instead of recomputing it a second time.
        if feature_cache is not None:
            prepare_examples(dataset, ["onset"], feature_cache, transform)

        self.examples: list[ValExample] = []
        for item in dataset:
            path = transform.apply(item.audio())
            if feature_cache is not None:
                # Keep only the path + small frame_times resident; the feature tensor
                # is loaded transiently per score (load_features), so the whole val
                # set's CQT never sits in memory at once.
                fpath = example_path(feature_cache, item.name, transform.name)
                frame_times = np.load(fpath)["frame_times"]
                features = None
            else:
                feats = extract_features(path)
                frame_times = feats.frame_times
                features = feats.values
                fpath = None
            self.examples.append(
                ValExample(
                    frame_times=frame_times,
                    contour=cached_f0(baseline, path, contour_cache),
                    notes=item.reference_notes(),
                    features=features,
                    feature_path=fpath,
                )
            )

    def score(self, segment: SegmentFn) -> dict[str, float]:
        return score_examples(segment, self.examples)