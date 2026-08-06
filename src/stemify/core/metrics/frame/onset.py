"""Onset-head quality, threshold-free: average precision over the onset curve.

The onset head emits a per-frame probability; turning it into discrete onsets needs
a peak-pick and a threshold, but that choice is a decoder concern and conflates two
questions -- how good is the curve, and where should the cut be. This metric answers
the first without the second, the onset analogue of ``VoicingAUROC``.

The metric is **average precision** (area under the precision-recall curve), not
AUROC, and the difference is deliberate. Onset frames are sparse -- a target frame
sits on ~1-2% of the grid -- and under that class imbalance ROC-AUC is optimistic:
the vast unvoiced majority makes the false-positive rate move slowly, so a curve can
look excellent while still firing far more false peaks than true ones. Average
precision has the positive class in its denominator (precision), so it degrades
honestly when the curve rings. That ringing *is* over-segmentation seen one stage
early -- a decoder cannot un-see a false onset the head is confident about.

Like ``VoicingAUROC`` the scalars also report the threshold that maximises frame-F1
and that F1, so the gap between the head's best operating point and the decoder's
default cut is visible. The PR curve itself goes in ``arrays`` for plotting.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from stemify.core.metrics.base import MetricResult
from stemify.core.metrics.frame.base import FrameMetric


class OnsetPR(FrameMetric):
    """Threshold-free onset-curve quality (average precision).

    ``true`` is the per-frame onset target (1 on/near a note onset, 0 elsewhere);
    ``pred`` is the head's per-frame onset probability. Both 1-D and frame-aligned
    on the same CQT grid -- these are a label and a probability, not Hz, which the
    ``FrameMetric`` contract allows (it requires only that the two are aligned)."""

    name = "onset_pr"

    def compute(self, true: np.ndarray, pred: np.ndarray) -> MetricResult:
        self._check(true, pred)
        labels = true.astype(bool)
        # Average precision is undefined without a positive frame (a track, or a
        # pool, with no onsets at all -- nothing to rank towards).
        if not labels.any():
            return MetricResult(self.name, {"onset_ap": float("nan")})
        best_thr, best_f1 = _best_f1_threshold(labels, pred)
        precision, recall, _ = precision_recall_curve(labels, pred)
        return MetricResult(
            self.name,
            scalars={
                "onset_ap": float(average_precision_score(labels, pred)),
                "onset_best_threshold": best_thr,
                "onset_best_f1": best_f1,
                "onset_positive_rate": float(labels.mean()),
            },
            arrays={"pr_precision": precision, "pr_recall": recall},
        )


def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """The probability cut that maximises frame-F1, and that F1 -- the head's best
    operating point, to compare the decoder's default peak-pick threshold against."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5, 0.0
    # precision/recall carry one extra entry (recall=0 with no threshold); pair each
    # threshold with its precision/recall before taking the F1-maximising cut.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
