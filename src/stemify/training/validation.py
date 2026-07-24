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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset
from stemify.evaluation.metrics import PRF, Fragmentation
from stemify.evaluation.metrics.frame import VoicingAUROC
from stemify.features import Features, extract_features
from stemify.notes import NoteEvent
from stemify.training.cache import cached_f0, example_path, features_npy
from stemify.training.examples import prepare_examples
from stemify.transforms import AudioTransform, Identity

# A segmentation step: cached features + their contour -> notes. The validator
# depends on this shape, not on any concrete encoder/decoder -- the caller wraps a
# checkpoint's NeuralEncoder + a Decoder and passes their composed step.
SegmentFn = Callable[[Contour, Features], list[NoteEvent]]
# The raw voicing step: cached features -> per-frame P(voiced), or None with no
# voicing head. Its threshold-free counterpart to SegmentFn, for the AUROC path.
VoicingProbFn = Callable[[Features], "np.ndarray | None"]


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
    voicing_target: np.ndarray | None = None  # (T,) gt voiced mask, for voicing AUROC

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
    given (typically a decoder over a checkpoint's ``NeuralEncoder.encode_features``,
    the same encoder/decoder pair deployment composes into a ``Segmenter``), so how
    notes are made lives entirely there and eval can't drift from deploy. Pure given
    ``examples`` and ``segment``."""
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


def score_voicing(
    voicing_prob: VoicingProbFn,
    examples: Sequence[ValExample],
) -> dict[str, float]:
    """Frame-level voicing-head quality (AUROC) over precomputed val examples.

    Pools every example's per-frame P(voiced) and its ground-truth voiced label
    (both on the CQT grid, so no resampling) and scores once -- threshold-free, so
    it measures the head independent of the 0.5 cut ``segment_features`` uses. Like
    ``score_examples`` it takes a callable, not a model, so the validator stays
    model-agnostic. Returns ``{}`` when the model has no voicing head or no example
    carries a label (nothing to score)."""
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for ex in examples:
        if ex.voicing_target is None:
            continue
        p = voicing_prob(ex.load_features())
        if p is None:  # onset-only model: no voicing head to score
            return {}
        labels.append(ex.voicing_target)
        probs.append(p)
    if not labels:
        return {}
    result = VoicingAUROC().calculate(np.concatenate(labels), np.concatenate(probs))
    return result.scalars


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
        # built instead of recomputing it a second time. Prepare the voicing target
        # too (cheap backfill), so the per-frame voiced label for the AUROC path is
        # in every condition's cache, not just the stem one the BCE loader built.
        if feature_cache is not None:
            prepare_examples(dataset, ["onset", "voicing"], feature_cache, transform)

        self.examples: list[ValExample] = []
        for item in dataset:
            path = transform.apply(item.audio())
            if feature_cache is not None:
                # Keep only the path + small frame_times/label resident; the feature
                # tensor is loaded transiently per score (load_features), so the whole
                # val set's CQT never sits in memory at once.
                fpath = example_path(feature_cache, item.name, transform.name)
                cached = np.load(fpath)
                frame_times = cached["frame_times"]
                voicing_target = (
                    cached["target_voicing"] if "target_voicing" in cached.files else None
                )
                features = None
            else:
                feats = extract_features(path)
                frame_times = feats.frame_times
                features = feats.values
                voicing_target = None
                fpath = None
            self.examples.append(
                ValExample(
                    frame_times=frame_times,
                    contour=cached_f0(baseline, path, contour_cache),
                    notes=item.reference_notes(),
                    features=features,
                    feature_path=fpath,
                    voicing_target=voicing_target,
                )
            )

    def score(self, segment: SegmentFn) -> dict[str, float]:
        return score_examples(segment, self.examples)

    def score_voicing(self, voicing_prob: VoicingProbFn) -> dict[str, float]:
        return score_voicing(voicing_prob, self.examples)