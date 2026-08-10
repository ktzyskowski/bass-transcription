"""Voicing-head quality, threshold-free: AUROC over per-frame P(voiced).

The voicing head emits a per-frame probability; turning it into a note/rest mask
needs a threshold (0.5 by default), but that choice is arbitrary and conflates two
questions -- how good is the head, and where should the cut be. AUROC answers the
first without the second: it is the probability the head ranks a random voiced
frame above a random unvoiced one, i.e. the area under the ROC swept over *every*
threshold. 1.0 is perfect ranking, 0.5 a coin flip -- and it is unchanged by where
0.5 happens to fall.

That invariance is also why ``RankingMetric`` range-checks its inputs: AUROC is
identical under any monotone squash, so raw logits would score exactly as well here
while making ``voicing_best_threshold`` a cut on a scale no decoder uses.

The ROC curve itself (fpr, tpr, thresholds) goes in ``arrays`` for plotting and
for reading off an operating point; the scalars also report the threshold that
maximises frame-F1 and that F1, so the gap between the optimal cut and the default
0.5 is visible at a glance.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from stemify.metrics.base import MetricResult
from stemify.metrics.frame.base import RankingMetric


def _empty() -> np.ndarray:
    return np.array([], dtype=float)


@dataclass(frozen=True)
class VoicingAUROCResult(MetricResult):
    """Voicing-head quality. Every field is always present -- see ``voicing_auroc``."""

    name = "voicing_auroc"
    SCALARS = ("voicing_auroc", "voicing_best_threshold", "voicing_best_f1")

    voicing_auroc: float
    """Area under the ROC, or ``nan`` when the pool is single-class.

    ``nan`` rather than an absent key, for the same reason as ``OnsetPR``: the
    training callback turns whatever keys a result carries into MLflow metric names,
    so a val pool with no rests (or no notes) used to end the two series below for
    that epoch rather than record that they were undefined.
    """

    voicing_best_threshold: float
    voicing_best_f1: float

    roc_fpr: np.ndarray = field(default_factory=_empty)
    roc_tpr: np.ndarray = field(default_factory=_empty)
    roc_thresholds: np.ndarray = field(default_factory=_empty)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "roc_fpr": self.roc_fpr,
            "roc_tpr": self.roc_tpr,
            "roc_thresholds": self.roc_thresholds,
        }


class VoicingAUROC(RankingMetric):
    """Threshold-free voicing quality.

    ``true`` is the per-frame voiced label (1 voiced, 0 rest); ``pred`` is the
    head's per-frame P(voiced), both on the same CQT grid.
    """

    name = "voicing_auroc"

    def compute(self, true: np.ndarray, pred: np.ndarray) -> VoicingAUROCResult:
        self._check(true, pred)
        labels = true.astype(bool)
        # AUROC is undefined without both classes present (all-voiced or all-rest).
        if labels.all() or (~labels).all():
            return VoicingAUROCResult(
                voicing_auroc=float("nan"),
                voicing_best_threshold=float("nan"),
                voicing_best_f1=float("nan"),
            )
        best_thr, best_f1 = self._best_f1_threshold(labels, pred)
        fpr, tpr, thresholds = roc_curve(labels, pred)
        return VoicingAUROCResult(
            voicing_auroc=float(roc_auc_score(labels, pred)),
            voicing_best_threshold=best_thr,
            voicing_best_f1=best_f1,
            roc_fpr=fpr,
            roc_tpr=tpr,
            roc_thresholds=thresholds,
        )
