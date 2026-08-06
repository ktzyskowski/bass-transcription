"""Frame-level metrics: score two frame-aligned Hz arrays.

``FrameMetric`` is the contract; the caller flattens every track's frames into
one pair and scores once (see ``stemify.evaluation.frame``).
"""

from stemify.core.metrics.frame.base import FrameMetric
from stemify.core.metrics.frame.confusion import ConfusionResult, PitchConfusion
from stemify.core.metrics.frame.onset import OnsetPR
from stemify.core.metrics.frame.pitch import PitchAccuracy, PitchAccuracyResult
from stemify.core.metrics.frame.scoring import AlignedPairs, score_cues
from stemify.core.metrics.frame.voicing import VoicingAUROC

__all__ = [
    "FrameMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
    "VoicingAUROC",
    "OnsetPR",
    "AlignedPairs",
    "score_cues",
]
