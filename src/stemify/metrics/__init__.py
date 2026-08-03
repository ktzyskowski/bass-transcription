"""Metrics: pure functions from a (reference, estimate) pair to numbers.

Two families, each in its own subpackage, split by input shape (see ``base``):

  - ``frame`` -- ``FrameMetric`` over two frame-aligned arrays. ``PitchAccuracy``,
    ``PitchConfusion`` (Hz), ``OnsetPR``, ``VoicingAUROC`` (label vs probability).
    Alignment and pooling happen before the metric, in the cues harness
    (``stemify.evaluation.cues``, using ``stemify.evaluation.alignment`` for pitch).
  - ``note`` -- ``NoteMetric`` over lists of per-track note lists. ``PRF``,
    ``Fragmentation``. Matching (``matched_pairs``) is shared within the subpackage.

Everything is re-exported here, so callers import from ``stemify.metrics``
regardless of which subpackage a metric lives in.
"""

from stemify.metrics.base import MetricResult
from stemify.metrics.frame import (
    ConfusionResult,
    FrameMetric,
    OnsetPR,
    PitchAccuracy,
    PitchAccuracyResult,
    PitchConfusion,
    AlignedPairs,
    VoicingAUROC,
    score_cues,
)
from stemify.metrics.note import (
    DEFAULT_NOTE_METRICS,
    PRF,
    Fragmentation,
    FragmentationResult,
    NoteMetric,
    PRFResult,
    count_matches,
    matched_pairs,
    prf_rates,
    score_notes,
)

__all__ = [
    "MetricResult",
    "FrameMetric",
    "NoteMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
    "OnsetPR",
    "VoicingAUROC",
    "AlignedPairs",
    "score_cues",
    "PRF",
    "PRFResult",
    "Fragmentation",
    "FragmentationResult",
    "DEFAULT_NOTE_METRICS",
    "score_notes",
    "prf_rates",
    "matched_pairs",
    "count_matches",
]