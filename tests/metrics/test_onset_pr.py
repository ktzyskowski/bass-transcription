"""OnsetPR: threshold-free onset-curve quality via average precision.

Synthetic targets/probs, so ranking quality, the imbalance the metric is chosen
for, and the degenerate no-onset case are checked without a model.
"""

import math

import numpy as np
import pytest

from stemify.metrics.frame import OnsetPR


def test_perfect_ranking_scores_one():
    true = np.array([0, 0, 0, 1, 1], dtype=float)
    pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9])  # every onset ranks above every non-onset
    result = OnsetPR().calculate(true, pred)
    assert result.scalars["onset_ap"] == pytest.approx(1.0)
    assert result.scalars["onset_best_f1"] == pytest.approx(1.0)


def test_ap_penalises_false_peaks_under_imbalance():
    # One true onset in a sea of rest, plus a confident false peak. AP is well below
    # 1.0 because precision collapses when the ringing peak outranks nothing real --
    # the property (over ROC-AUC) this metric is chosen for.
    true = np.zeros(100, dtype=float)
    true[50] = 1.0
    pred = np.full(100, 0.01)
    pred[50] = 0.9   # the true onset
    pred[10] = 0.95  # a confident false peak, ranked above the true one
    ap = OnsetPR().calculate(true, pred).scalars["onset_ap"]
    assert ap < 0.6  # a false peak above the true onset tanks average precision


def test_positive_rate_is_reported():
    true = np.array([0, 0, 0, 0, 1], dtype=float)
    result = OnsetPR().calculate(true, np.linspace(0, 1, 5))
    assert result.scalars["onset_positive_rate"] == pytest.approx(0.2)


def test_no_onsets_is_nan():
    # An all-rest pool -> AP undefined; NaN rather than a misleading number.
    true = np.zeros(5, dtype=float)
    ap = OnsetPR().calculate(true, np.linspace(0, 1, 5)).scalars["onset_ap"]
    assert math.isnan(ap)


def test_pr_curve_is_returned_for_plotting():
    true = np.array([0, 0, 1, 1], dtype=float)
    result = OnsetPR().calculate(true, np.array([0.2, 0.4, 0.6, 0.8]))
    assert {"pr_precision", "pr_recall"} <= set(result.arrays)
