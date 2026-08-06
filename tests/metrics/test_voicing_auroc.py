"""VoicingAUROC: threshold-free voicing quality from per-frame labels + probs.

Synthetic labels/probs, so ranking quality and the degenerate single-class case
are checked without a model.
"""

import math

import numpy as np
import pytest

from stemify.core.metrics.frame import VoicingAUROC


def test_perfect_ranking_scores_one():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # every voiced ranks above rest
    result = VoicingAUROC().compute(labels, probs)
    assert result.scalars["voicing_auroc"] == 1.0
    # a threshold between 0.3 and 0.7 separates them -> frame-F1 of 1.0
    assert result.scalars["voicing_best_f1"] == pytest.approx(1.0)
    assert 0.3 < result.scalars["voicing_best_threshold"] <= 0.7


def test_inverted_ranking_scores_zero():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # voiced ranked below rest
    assert VoicingAUROC().compute(labels, probs).scalars["voicing_auroc"] == 0.0


def test_auroc_is_threshold_free():
    # Same ranking, probabilities squashed far from 0.5: AUROC unchanged because it
    # depends only on order, not on where any threshold falls.
    labels = np.array([0, 0, 1, 1], dtype=float)
    a = VoicingAUROC().compute(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    b = VoicingAUROC().compute(labels, np.array([0.01, 0.02, 0.51, 0.52]))
    assert a.scalars["voicing_auroc"] == b.scalars["voicing_auroc"] == 1.0


def test_single_class_is_nan():
    # No rest frames -> AUROC undefined; report NaN rather than a misleading number.
    labels = np.ones(5, dtype=float)
    auroc = VoicingAUROC().compute(labels, np.linspace(0, 1, 5)).scalars["voicing_auroc"]
    assert math.isnan(auroc)


def test_roc_curve_is_returned_for_plotting():
    labels = np.array([0, 0, 1, 1], dtype=float)
    result = VoicingAUROC().compute(labels, np.array([0.2, 0.4, 0.6, 0.8]))
    assert {"roc_fpr", "roc_tpr", "roc_thresholds"} <= set(result.arrays)
    assert result.arrays["roc_fpr"].shape == result.arrays["roc_tpr"].shape
