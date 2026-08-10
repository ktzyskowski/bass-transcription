"""Precision / recall / F1 over note events -- the metric a tab is graded on.

Kept as one metric, not three: precision, recall and F1 all fall out of the same
tp/fp/fn, so there is one match per track and one place the counts live. Counts
are summed across tracks, then the rates are derived once from the totals --
per-track F1 cannot be averaged.

The per-track counts the pooled sum is built from are not thrown away: they go into
``per_item`` (one ``Counts`` per input track, in input order). The worst-item ranking
a harness shows is then a projection of that single match -- no second scoring pass,
and pooled + per-item can never disagree.

**The result *is* the pooled counts.** ``PRFResult`` inherits ``Counts``, so the
headline numbers and every per-track row are the same type and derive their rates
through the same three properties. There is no ``prf_rates`` helper any more, because
there is no second way to turn counts into rates for it to be.
"""

from dataclasses import dataclass

from stemify.metrics.base import MetricResult
from stemify.metrics.note.base import Counts, NoteMetric, count_matches
from stemify.music.notes import NoteEvent


@dataclass(frozen=True)
class PRFResult(MetricResult, Counts):
    name = "note-prf"
    SCALARS = ("tp", "fp", "fn", "precision", "recall", "f1")

    per_item: tuple[Counts, ...] = ()
    """Per-track counts, in input order -- the rows the pooled totals were summed from.

    Carried on the result rather than stuffed into ``arrays`` as an ``(n, 3)`` integer
    block a caller had to know the column order of.
    """

    def render(self) -> str:
        return "\n".join([
            f"  reference / estimated notes  "
            f"{self.n_reference:,} / {self.n_estimated:,}",
            f"  precision                    {self.precision:>7.1%}",
            f"  recall                       {self.recall:>7.1%}",
            f"  F1                           {self.f1:>7.1%}",
            f"  tp / fp / fn                 {self.tp:,} / {self.fp:,} / {self.fn:,}",
        ])


class PRF(NoteMetric):
    name = "note-prf"

    def compute(
        self,
        references: list[list[NoteEvent]],
        estimates: list[list[NoteEvent]],
    ) -> PRFResult:
        per_item = tuple(
            count_matches(ref, est, self.onset_tolerance, self.pitch_tolerance)
            for ref, est in zip(references, estimates, strict=True)
        )
        return PRFResult(
            tp=sum(c.tp for c in per_item),
            fp=sum(c.fp for c in per_item),
            fn=sum(c.fn for c in per_item),
            per_item=per_item,
        )
