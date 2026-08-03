"""Precision / recall / F1 over note events -- the metric a tab is graded on.

Kept as one metric, not three: precision, recall and F1 all fall out of the same
tp/fp/fn, so there is one match per track and one place the counts live. Counts
are summed across tracks, then the rates are derived once from the totals --
per-track F1 cannot be averaged.

The per-track counts the pooled sum is built from are not thrown away: they go into
``arrays["per_item_counts"]`` (one ``(tp, fp, fn)`` row per input track, in input
order). The worst-item ranking a harness shows is then a projection of that single
match -- no second scoring pass, and pooled + per-item can never disagree.
"""

import numpy as np

from stemify.metrics.base import MetricResult
from stemify.metrics.note.base import NoteMetric, count_matches
from stemify.notes import NoteEvent


def prf_rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    """P/R/F plus the raw counts, from one track's or the pool's ``(tp, fp, fn)``.

    The one place the rate formulas live, so a per-track row and the pooled totals are
    derived identically (rates are never averaged across tracks -- they are recomputed
    from counts, which is why the counts, not the rates, are what gets summed/stored)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


class PRFResult(MetricResult):
    def render(self) -> str:
        s = self.scalars
        return "\n".join([
            f"  reference / estimated notes  "
            f"{int(s['tp'] + s['fn']):,} / {int(s['tp'] + s['fp']):,}",
            f"  precision                    {s['precision']:>7.1%}",
            f"  recall                       {s['recall']:>7.1%}",
            f"  F1                           {s['f1']:>7.1%}",
            f"  tp / fp / fn                 "
            f"{int(s['tp']):,} / {int(s['fp']):,} / {int(s['fn']):,}",
        ])


class PRF(NoteMetric):
    name = "note-prf"

    def compute(
        self,
        references: list[list[NoteEvent]],
        estimates: list[list[NoteEvent]],
    ) -> MetricResult:
        counts = [
            count_matches(ref, est, self.onset_tolerance, self.pitch_tolerance)
            for ref, est in zip(references, estimates)
        ]
        per_item = np.array(counts, dtype=int).reshape(-1, 3)  # (n_tracks, 3): tp/fp/fn
        tp, fp, fn = (int(per_item[:, i].sum()) for i in range(3))
        return PRFResult(
            self.name,
            prf_rates(tp, fp, fn),
            arrays={"per_item_counts": per_item},
        )