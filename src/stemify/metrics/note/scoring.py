"""The note-family scorer: run note metrics over per-track reference/estimate lists.

The peer of ``metrics.frame.scoring`` (frame-channel pooling), for the note family.
Note metrics self-aggregate -- each loops tracks and sums counts internally -- so
"scoring" here is thin: apply each metric to the nested lists, and (optionally) score
each track alone for the worst-item breakdown. It earns its own module anyway because
two callers need exactly this and must not drift: the offline ``NoteEvaluation`` and
the in-training async validator (``training.validation.score_examples``). Keeping the
metric-calling in one leaf means a tolerance or a default-metric change lands in both.

Pure over the note lists -- no dataset, no model -- so how the estimates were produced
(a whole ``Segmenter`` offline, a checkpoint's encoder+decoder in training) stays with
the caller.
"""

from collections.abc import Sequence

from stemify.metrics.base import MetricResult
from stemify.metrics.note.fragmentation import Fragmentation
from stemify.metrics.note.prf import PRF

Notes = list  # a per-track list of NoteEvent; kept loose to avoid importing the type

# PRF is the headline (what a tab is graded on); Fragmentation explains the false
# positives PRF only counts. Both read the same match, so they agree on which
# estimates were wrong.
DEFAULT_NOTE_METRICS = (PRF(), Fragmentation())


def score_notes(
    references: Sequence[Notes],
    estimates: Sequence[Notes],
    metrics=DEFAULT_NOTE_METRICS,
) -> list[MetricResult]:
    """Each metric's result over the whole set of per-track (reference, estimate) lists."""
    return [m.calculate(list(references), list(estimates)) for m in metrics]


def per_item_prf(
    names: Sequence[str],
    references: Sequence[Notes],
    estimates: Sequence[Notes],
    ranker: PRF | None = None,
) -> list[tuple[str, dict[str, float]]]:
    """``(name, PRF scalars)`` per track -- the worst-item breakdown a pooled F1 hides.

    ``ranker`` reuses a ``PRF`` from the scored metrics so per-item tolerances match
    the pooled numbers; a default is used if the caller has none."""
    ranker = ranker or PRF()
    return [
        (name, ranker.calculate([ref], [est]).scalars)
        for name, ref, est in zip(names, references, estimates)
    ]
