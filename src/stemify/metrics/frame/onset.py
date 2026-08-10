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

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from stemify.metrics.base import MetricResult
from stemify.metrics.frame.base import RankingMetric


def _empty() -> np.ndarray:
    return np.array([], dtype=float)


@dataclass(frozen=True)
class OnsetPRResult(MetricResult):
    """Onset-curve quality. Every field is always present -- see ``onset_ap``."""

    name = "onset_pr"
    SCALARS = (
        "onset_ap",
        "onset_best_threshold",
        "onset_best_f1",
        "onset_positive_rate",
    )

    onset_ap: float
    """Average precision, or ``nan`` when the pool holds no onset at all.

    ``nan`` rather than an absent key. This branch used to return a one-entry result,
    dropping the three fields below -- and the training callback iterates whatever keys
    a result happens to have straight into MLflow metric names, so an onset-free pool
    ended the other three series for that epoch instead of recording that they could
    not be computed. A field cannot be dropped; a dict entry can.
    """

    onset_best_threshold: float
    onset_best_f1: float
    onset_positive_rate: float
    """Fraction of frames carrying an onset target. Real even on the degenerate path
    (it is 0.0, which is the fact that explains the ``nan`` beside it)."""

    pr_precision: np.ndarray = field(default_factory=_empty)
    pr_recall: np.ndarray = field(default_factory=_empty)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {"pr_precision": self.pr_precision, "pr_recall": self.pr_recall}


class OnsetPR(RankingMetric):
    """Threshold-free onset-curve quality (average precision).

    ``true`` is the per-frame onset target (1 on/near a note onset, 0 elsewhere);
    ``pred`` is the head's per-frame onset probability, both on the same CQT grid.
    """

    name = "onset_pr"

    def compute(self, true: np.ndarray, pred: np.ndarray) -> OnsetPRResult:
        self._check(true, pred)
        labels = true.astype(bool)
        positive_rate = float(labels.mean()) if labels.size else 0.0
        # Average precision is undefined without a positive frame (a track, or a
        # pool, with no onsets at all -- nothing to rank towards).
        if not labels.any():
            return OnsetPRResult(
                onset_ap=float("nan"),
                onset_best_threshold=float("nan"),
                onset_best_f1=float("nan"),
                onset_positive_rate=positive_rate,
            )
        best_thr, best_f1 = self._best_f1_threshold(labels, pred)
        precision, recall, _ = precision_recall_curve(labels, pred)
        return OnsetPRResult(
            onset_ap=float(average_precision_score(labels, pred)),
            onset_best_threshold=best_thr,
            onset_best_f1=best_f1,
            onset_positive_rate=positive_rate,
            pr_precision=precision,
            pr_recall=recall,
        )
