"""Metrics: pure functions from an aligned (reference, estimate) pair to numbers.

See ``base`` for the contract. Alignment and aggregation live in
``stemify.evaluation.alignment``, deliberately outside this package.
"""

from stemify.evaluation.metrics.base import Metric, MetricResult
from stemify.evaluation.metrics.confusion import PitchConfusion
from stemify.evaluation.metrics.pitch import PitchAccuracy

__all__ = ["Metric", "MetricResult", "PitchAccuracy", "PitchConfusion"]
