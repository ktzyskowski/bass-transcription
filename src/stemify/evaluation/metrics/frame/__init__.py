"""Frame-level metrics: score two frame-aligned Hz arrays.

``FrameMetric`` is the contract; the caller flattens every track's frames into
one pair and scores once (see ``stemify.evaluation.frame``).
"""

from stemify.evaluation.metrics.frame.base import FrameMetric
from stemify.evaluation.metrics.frame.confusion import ConfusionResult, PitchConfusion
from stemify.evaluation.metrics.frame.pitch import PitchAccuracy, PitchAccuracyResult

__all__ = [
    "FrameMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
]