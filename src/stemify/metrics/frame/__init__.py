"""Frame-level metrics: score two frame-aligned arrays.

``FrameMetric`` is the contract, split into ``PitchMetric`` (Hz, 0.0 unvoiced) and
``RankingMetric`` (a target and a probability); the caller flattens every track's
frames into one pair and scores once (see ``stemify.evaluation.cues``).
"""

from stemify.metrics.frame.base import FrameMetric, PitchMetric, RankingMetric
from stemify.metrics.frame.confusion import ConfusionResult, PitchConfusion
from stemify.metrics.frame.onset import OnsetPR, OnsetPRResult
from stemify.metrics.frame.pitch import PitchAccuracy, PitchAccuracyResult
from stemify.metrics.frame.scoring import AlignedPairs, score_cues
from stemify.metrics.frame.voicing import VoicingAUROC, VoicingAUROCResult

__all__ = [
    "FrameMetric",
    "PitchMetric",
    "RankingMetric",
    "PitchAccuracy",
    "PitchAccuracyResult",
    "PitchConfusion",
    "ConfusionResult",
    "VoicingAUROC",
    "VoicingAUROCResult",
    "OnsetPR",
    "OnsetPRResult",
    "AlignedPairs",
    "score_cues",
]
