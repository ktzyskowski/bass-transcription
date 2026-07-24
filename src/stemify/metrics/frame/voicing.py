"""Voicing-head quality, threshold-free: AUROC over per-frame P(voiced).

The voicing head emits a per-frame probability; turning it into a note/rest mask
needs a threshold (0.5 by default), but that choice is arbitrary and conflates two
questions -- how good is the head, and where should the cut be. AUROC answers the
first without the second: it is the probability the head ranks a random voiced
frame above a random unvoiced one, i.e. the area under the ROC swept over *every*
threshold. 1.0 is perfect ranking, 0.5 a coin flip -- and it is unchanged by where
0.5 happens to fall.

The ROC curve itself (fpr, tpr, thresholds) goes in ``arrays`` for plotting and
for reading off an operating point; the scalars also report the threshold that
maximises frame-F1 and that F1, so the gap between the optimal cut and the default
0.5 is visible at a glance.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from stemify.metrics.base import MetricResult
from stemify.metrics.frame.base import FrameMetric


class VoicingAUROC(FrameMetric):
    """Threshold-free voicing quality.

    ``true`` is the per-frame voiced label (1 voiced, 0 rest); ``pred`` is the
    head's per-frame P(voiced). Both 1-D and aligned. Unlike the pitch metrics
    these are labels and probabilities, not Hz -- the FrameMetric contract is only
    that the two arrays are frame-aligned, which they are (same CQT grid)."""

    name = "voicing_auroc"

    def calculate(self, true: np.ndarray, pred: np.ndarray) -> MetricResult:
        self._check(true, pred)
        labels = true.astype(bool)
        # AUROC is undefined without both classes present (all-voiced or all-rest).
        if labels.all() or (~labels).all():
            return MetricResult(self.name, {"voicing_auroc": float("nan")})
        best_thr, best_f1 = _best_f1_threshold(labels, pred)
        fpr, tpr, thresholds = roc_curve(labels, pred)
        return MetricResult(
            self.name,
            scalars={
                "voicing_auroc": float(roc_auc_score(labels, pred)),
                "voicing_best_threshold": best_thr,
                "voicing_best_f1": best_f1,
            },
            arrays={"roc_fpr": fpr, "roc_tpr": tpr, "roc_thresholds": thresholds},
        )


def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """The probability cut that maximises frame-F1, and that F1 -- the operating
    point to compare the default 0.5 against."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5, 0.0
    # precision/recall have one more entry than thresholds (the final recall=0
    # point has no threshold); pair each threshold with its precision/recall.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
