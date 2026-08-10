"""Metrics: pure functions from a (reference, estimate) pair to numbers.

Two families, each in its own subpackage, split by input shape (see ``base``):

  - ``frame`` -- ``FrameMetric`` over two frame-aligned arrays, itself split by what
    those arrays hold: ``PitchMetric`` (Hz, 0.0 unvoiced -- ``PitchAccuracy``,
    ``PitchConfusion``) and ``RankingMetric`` (a target and a probability --
    ``OnsetPR``, ``VoicingAUROC``). Alignment and pooling happen before the metric,
    in the cues harness (``stemify.evaluation.cues``, using
    ``stemify.evaluation.alignment`` for pitch).
  - ``note`` -- ``NoteMetric`` over lists of per-track note lists. ``PRF``,
    ``Fragmentation``. Matching (``matched_pairs``) and the ``Counts`` every number
    is derived from are shared within the subpackage.

Every result is a frozen dataclass whose fields are its numbers and whose ``SCALARS``
names its public record, so the set of keys a run reports is a property of the types
rather than of the data -- see ``base`` for why that matters to MLflow.

Everything is re-exported here, so callers import from ``stemify.metrics``
regardless of which subpackage a metric lives in.
"""

from stemify.metrics.base import MetricResult
from stemify.metrics.frame import (
    AlignedPairs,
    ConfusionResult,
    FrameMetric,
    OnsetPR,
    OnsetPRResult,
    PitchAccuracy,
    PitchAccuracyResult,
    PitchConfusion,
    PitchMetric,
    RankingMetric,
    VoicingAUROC,
    VoicingAUROCResult,
    score_cues,
)
from stemify.metrics.note import (
    DEFAULT_NOTE_METRICS,
    PRF,
    Counts,
    Fragmentation,
    FragmentationResult,
    NoteMetric,
    PRFResult,
    count_matches,
    matched_pairs,
    score_notes,
)

__all__ = [
    "MetricResult",
    "FrameMetric",
    "PitchMetric",
    "RankingMetric",
    "NoteMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
    "OnsetPR",
    "OnsetPRResult",
    "VoicingAUROC",
    "VoicingAUROCResult",
    "AlignedPairs",
    "score_cues",
    "PRF",
    "PRFResult",
    "Counts",
    "Fragmentation",
    "FragmentationResult",
    "DEFAULT_NOTE_METRICS",
    "score_notes",
    "matched_pairs",
    "count_matches",
]
