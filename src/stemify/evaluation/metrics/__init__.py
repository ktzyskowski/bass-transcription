"""Metrics: pure functions from a (reference, estimate) pair to numbers.

Two families, each in its own subpackage, split by input shape (see ``base``):

  - ``frame`` -- ``FrameMetric`` over two frame-aligned Hz arrays.
    ``PitchAccuracy``, ``PitchConfusion``. Alignment and pooling happen before
    the metric, in ``stemify.evaluation.alignment`` / the frame harness.
  - ``note`` -- ``NoteMetric`` over lists of per-track note lists. ``PRF``,
    ``OverSegmentation``. Matching (``matched_pairs``) is shared within the subpackage.

Everything is re-exported here, so callers import from ``stemify.evaluation.metrics``
regardless of which subpackage a metric lives in.
"""

from stemify.evaluation.metrics.base import MetricResult
from stemify.evaluation.metrics.frame import (
    ConfusionResult,
    FrameMetric,
    PitchAccuracy,
    PitchAccuracyResult,
    PitchConfusion,
)
from stemify.evaluation.metrics.note import (
    PRF,
    NoteMetric,
    OverSegmentation,
    OverSegResult,
    PRFResult,
    count_matches,
    matched_pairs,
)

__all__ = [
    "MetricResult",
    "FrameMetric",
    "NoteMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
    "PRF",
    "PRFResult",
    "OverSegmentation",
    "OverSegResult",
    "matched_pairs",
    "count_matches",
]