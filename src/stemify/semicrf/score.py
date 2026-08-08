"""Score of *given* segmentations under the scorer.

The third way to score a tiling, alongside ``forward`` (logsumexp over all of them)
and ``viterbi`` (the best one). The NLL is ``log Z - score(gold)``, so this is the gold
half -- but nothing here knows the segmentation is gold, only that it is a list of
``(label, i, j)`` spans.

Separate from ``gold.py`` on purpose: that module decides *which* segmentation is gold
(the audible-gap rule, pure numpy) and changes when the labelling rule changes; this one
turns any segmentation into a number and changes when the *scorer* does. Different
reasons to edit, so different files.

It calls the same ``SpanScorer.unary`` the recurrence does. That is the point: while
this module computed the potential itself, ``log Z`` and ``score(gold)`` could drift
apart, and a drift is invisible -- the NLL still falls, gradients still flow, and the
objective simply stops being a likelihood.
"""

import torch

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.spans import Grid, SpanScorer

# One segment: (label, start frame, end frame), end exclusive.
Segmentation = list[tuple[int, int, int]]


def _prev_labels(lab: torch.Tensor, first: torch.Tensor, start: int) -> torch.Tensor:
    """The label preceding each segment, ``START`` wherever ``first`` is set.

    ``start`` is passed rather than imported: it is the *scorer's* ``START`` row, which
    sits one past its alphabet and so moves with the alphabet's width. Reading the
    two-label constant here would index a pitch label instead on any wider scorer --
    silently, since it is a valid row, and visibly only in the first segment of each
    track."""
    prev = torch.cat([lab.new_full((1,), start), lab[:-1]])
    return torch.where(first, torch.full_like(prev, start), prev)


def segmentation_scores(
    scorer: SpanScorer,
    ev: Evidence,
    per_track: list[Segmentation],
    grid: Grid,
) -> torch.Tensor:
    """``(B,)`` unnormalised scores for a batch's segmentations, in one vectorised pass.

    Scoring one track at a time in a Python loop was the single largest cost in a
    training step -- a batch's worth of segments is thousands of tiny kernel launches.
    Flattening every track's segments into one indexed set makes the whole batch a fixed
    handful of ops, and ``index_add`` folds them back per track.

    Indices are never clamped, so a caller passing a span past a track's true length
    would silently read the flat padding ``build_evidence`` leaves there.
    """
    rows = [(b, *seg) for b, segs in enumerate(per_track) for seg in segs]
    out = ev.onset_logit.new_zeros(len(per_track))
    if not rows:
        return out
    t = torch.tensor(rows, dtype=torch.long, device=ev.device)   # (S, 4)
    b, lab, i, j = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

    # A track's first segment follows START; `rows` is grouped by track, so a change in
    # `b` marks each boundary.
    first = torch.ones_like(b, dtype=torch.bool)
    first[1:] = b[1:] != b[:-1]

    both = scorer.unary(ev, grid, b, i, j)                       # (S, 2)
    unary = both.gather(-1, lab.unsqueeze(-1)).squeeze(-1)       # (S,)
    prev = _prev_labels(lab, first, scorer.labels.START)
    terms = unary + scorer.transitions()[prev, lab]
    return out.index_add(0, b, terms)


def segmentation_score(
    scorer: SpanScorer, ev: Evidence, segs: Segmentation, grid: Grid
) -> torch.Tensor:
    """Unnormalised score of one track's segmentation -- the batch of one."""
    return segmentation_scores(scorer, ev, [segs], grid)[0]
