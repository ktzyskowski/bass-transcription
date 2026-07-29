"""Cached validation for the async validator: note-level and cues-level.

The async eval is dominated by pyin, but pyin's pitch contour is fixed across
checkpoints -- only the onset/voicing model changes. ``OnsetValidator`` runs pyin
(and the CQT) over the val tracks *once* at construction and holds the results, so
each subsequent score is just a model forward and the scoring -- seconds, not minutes.
That is what makes async validation every few epochs cheap enough to leave on.

From that one forward per track it scores both altitudes: ``score_examples`` runs the
decoder for note-level metrics, ``score_cues_examples`` scores the encoder's cues
(onset-curve and voicing-head quality). Both are pure cores (given precomputed
examples) that delegate to the shared ``metrics`` scorers -- ``score_notes`` and
``score_cues`` -- so the in-training numbers cannot drift from the offline harness,
and both are tested with synthetic inputs and a fake model, no audio.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.data.base import Dataset
from stemify.features import FrameFeatures, extract_features
from stemify.metrics import MetricResult, ReferenceAlignedCues, score_cues, score_notes
from stemify.notes import NoteEvent
from stemify.pitch import PyinPitch
from stemify.serde import features_npy
from stemify.cache import Cache
from stemify.training.examples import prepare_examples
from stemify.training.labels import onset_targets
from stemify.transforms import AudioTransform, Identity

# A segmentation step: cached features + their contour -> notes. The validator
# depends on this shape, not on any concrete encoder/decoder -- the caller wraps a
# checkpoint's NeuralEncoder + a Decoder and passes their composed step.
SegmentFn = Callable[[Contour, FrameFeatures], list[NoteEvent]]
# The encode step: cached features + their contour -> the encoder's Cues. Used for
# cues-level scoring (onset + voicing head quality) -- a checkpoint's
# NeuralEncoder.encode_features, so the validator stays model-agnostic.
EncodeFn = Callable[[Contour, FrameFeatures], Cues]


@dataclass
class ValExample:
    """One val track, everything fixed across checkpoints precomputed.

    FrameFeatures are held either in memory (``features``, e.g. tests or the no-cache
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

    def load_features(self) -> FrameFeatures:
        """The track's CQT paired with its frame grid, from memory or disk. The disk
        read is transient -- freed after each score -- so memory stays flat over the
        run. ``feature_path`` is the TrackExample npz; the CQT is its .npy sidecar."""
        values = self.features
        if values is None:
            values = np.load(features_npy(self.feature_path))
        return FrameFeatures(values, self.frame_times)


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
    # Score through the shared note core, so the val numbers use the same metrics and
    # tolerances the offline NoteEvaluation reports. Flatten to the MLflow-friendly
    # note_* keys the callback logs.
    scalars = {r.name: r.scalars for r in score_notes(refs, ests)}
    prf, frag = scalars["note-prf"], scalars["fragmentation"]
    return {
        "note_f1": prf["f1"],
        "note_precision": prf["precision"],
        "note_recall": prf["recall"],
        "fragmentation": float(frag["fragmentation"]),
    }


def score_cues_examples(
    encode: EncodeFn,
    examples: Sequence[ValExample],
) -> list[MetricResult]:
    """Cues-level (encoder) quality over precomputed val examples: onset-curve and
    voicing-head, both threshold-free, in one model pass per track.

    For each example, encode its cached features + contour into ``Cues`` and turn
    that into the aligned per-channel pairs ``score_cues`` pools: onset (curve vs an
    impulse target built from the reference note onsets on the CQT grid) and voicing
    (P(voiced) vs the cached voiced label). Pitch is *not* scored here -- pyin pitch
    does not move between checkpoints, so it belongs to the offline ``CuesEvaluation``,
    not the per-epoch loop. Voicing drops out for an onset-only model. Like
    ``score_examples`` this takes a callable, not a model, so the validator stays
    model-agnostic; the shared ``metrics.frame.scoring.score_cues`` keeps it from drifting from
    the offline harness."""
    tracks: list[ReferenceAlignedCues] = []
    for ex in examples:
        cues = encode(ex.contour, ex.load_features())
        onset_true = onset_targets(
            np.array([n.onset for n in ex.notes], dtype=float), cues.frame_times
        )
        voicing = None
        if cues.voicing is not None and ex.voicing_target is not None:
            voicing = (cues.voicing, ex.voicing_target)
        tracks.append(ReferenceAlignedCues(onset=(cues.onset, onset_true), voicing=voicing))
    return score_cues(tracks)


class OnsetValidator:
    """Precompute pyin contours + CQT + reference notes once; score models cheaply."""

    def __init__(
        self,
        dataset: Dataset,
        transform: AudioTransform | None = None,
        cache: Cache | None = None,
    ) -> None:
        transform = transform or Identity()
        estimator = PyinPitch()

        # One ``cache`` replaces the old separate ``contour_cache``/``feature_cache``
        # paths, and unifies what their two ``None``s meant: with a cache, both the
        # CQT and the contour are disk-backed; without one, features are held in
        # memory and pyin reruns per launch (the test/small-set path).
        #
        # The CQT is the same artifact the training loader caches. Since features are
        # keyed on the resolved audio path, *every* condition now reuses whatever the
        # loader already built -- not just the stem tag, which was previously a
        # special case worth a comment. Prepare the voicing target too (a cheap
        # backfill, no CQT recompute), so the per-frame voiced label the AUROC path
        # needs is present for every condition.
        if cache is not None:
            prepare_examples(dataset, ["onset", "voicing"], cache, transform)

        self.examples: list[ValExample] = []
        for item in dataset:
            path = transform.apply(item.audio())
            if cache is not None:
                # Keep only the path + small frame_times/label resident; the feature
                # tensor is loaded transiently per score (load_features), so the whole
                # val set's CQT never sits in memory at once.
                fpath = cache.feature_path(path)
                cached = np.load(fpath)
                frame_times = cached["frame_times"]
                voicing_target = (
                    cached["target_voicing"] if "target_voicing" in cached.files else None
                )
                features = None
                contour = cache.contour(estimator, path)
            else:
                feats = extract_features(path)
                frame_times = feats.frame_times
                features = feats.values
                voicing_target = None
                fpath = None
                contour = estimator.estimate(path)
            self.examples.append(
                ValExample(
                    frame_times=frame_times,
                    contour=contour,
                    notes=item.reference_notes(),
                    features=features,
                    feature_path=fpath,
                    voicing_target=voicing_target,
                )
            )

    def score(self, segment: SegmentFn) -> dict[str, float]:
        return score_examples(segment, self.examples)

    def score_cues(self, encode: EncodeFn) -> list[MetricResult]:
        return score_cues_examples(encode, self.examples)