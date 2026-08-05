"""Frame-level metrics: score two frame-aligned Hz arrays.

``FrameMetric`` is the contract; the caller flattens every track's frames into
one pair and scores once (see ``stemify.evaluation.frame``).
"""

from stemify.metrics.frame.base import FrameMetric
from stemify.metrics.frame.confusion import ConfusionResult, PitchConfusion
from stemify.metrics.frame.onset import OnsetPR
from stemify.metrics.frame.pitch import PitchAccuracy, PitchAccuracyResult
from stemify.metrics.frame.scoring import AlignedPairs, score_cues
from stemify.metrics.frame.voicing import VoicingAUROC

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
