"""Precision / recall / F1 over note events -- the metric a tab is graded on.

Kept as one metric, not three: precision, recall and F1 all fall out of the same
tp/fp/fn, so there is one match per track and one place the counts live. Counts
are summed across tracks, then the rates are derived once from the totals --
per-track F1 cannot be averaged.
"""

from stemify.evaluation.metrics.base import MetricResult
from stemify.evaluation.metrics.note.base import NoteMetric, count_matches
from stemify.notes import NoteEvent


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

    def calculate(
        self,
        references: list[list[NoteEvent]],
        estimates: list[list[NoteEvent]],
    ) -> MetricResult:
        tp = fp = fn = 0
        for ref, est in zip(references, estimates):
            t, f, n = count_matches(ref, est, self.onset_tolerance, self.pitch_tolerance)
            tp += t
            fp += f
            fn += n
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return PRFResult(
            self.name,
            {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            },
        )