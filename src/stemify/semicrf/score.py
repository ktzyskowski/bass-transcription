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

from collections.abc import Sequence

import torch

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.spans import Grid, SpanScorer

# One segment: (label, start frame, end frame), end exclusive.
Segmentation = list[tuple[int, int, int]]


def _prev_labels(
    lab: torch.Tensor, first: torch.Tensor, b: torch.Tensor, seeds: torch.Tensor
) -> torch.Tensor:
    """The label preceding each segment; ``seeds[row]`` wherever ``first`` is set.

    Within a row the predecessor is just the segment before it, because ``rows`` is
    grouped by track and ordered. What needs saying is what precedes the *first* one,
    and that is not a constant: over a whole track it is ``START``, but over a chunk
    cut out of a track it is whatever segment the previous chunk ended on -- a real
    ``note`` or ``rest``, carrying a real transition.

    Assuming ``START`` there was a live defect. ``log Z`` reads the carried alpha and
    applies the true transition, so the two halves of the NLL scored different
    quantities and the chunked objective stopped being the whole-track one the moment
    the transition table moved off its zero init. Measured at 5.18 nats over a
    six-chunk track, and invisible at initialisation -- which is exactly where the
    telescoping tests look."""
    prev = torch.cat([lab.new_zeros(1), lab[:-1]])
    return torch.where(first, seeds[b], prev)


def segmentation_scores(
    scorer: SpanScorer,
    ev: Evidence,
    per_track: list[Segmentation],
    grid: Grid,
    prev_labels: Sequence[int] | None = None,
) -> torch.Tensor:
    """``(B,)`` unnormalised scores for a batch's segmentations, in one vectorised pass.

    Scoring one track at a time in a Python loop was the single largest cost in a
    training step -- a batch's worth of segments is thousands of tiny kernel launches.
    Flattening every track's segments into one indexed set makes the whole batch a fixed
    handful of ops, and ``index_add`` folds them back per track.

    ``prev_labels`` is the label preceding each row's *first* segment, one per row.
    ``None`` means ``START`` for every row, which is right for a whole track and wrong
    for a chunk cut out of one -- see ``_prev_labels``. It is a caller's argument rather
    than something derived here because only the dataset knows what the previous chunk
    ended on.

    Indices are never clamped, so a caller passing a span past a track's true length
    would silently read the flat padding ``build_evidence`` leaves there.
    """
    rows = [(b, *seg) for b, segs in enumerate(per_track) for seg in segs]
    out = ev.onset_logit.new_zeros(len(per_track))
    if not rows:
        return out
    t = torch.tensor(rows, dtype=torch.long, device=ev.device)   # (S, 4)
    b, lab, i, j = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

    # `rows` is grouped by track, so a change in `b` marks each row's first segment.
    first = torch.ones_like(b, dtype=torch.bool)
    first[1:] = b[1:] != b[:-1]
    # `START` is the *scorer's* row, one past its alphabet, so it moves with the
    # alphabet's width. Reading a two-label constant here would index a pitch label
    # instead -- silently, since it is a valid row.
    seeds = (
        torch.full((len(per_track),), scorer.labels.START,
                   dtype=torch.long, device=ev.device)
        if prev_labels is None
        else torch.tensor(list(prev_labels), dtype=torch.long, device=ev.device)
    )
    if seeds.shape[0] != len(per_track):
        raise ValueError(
            f"prev_labels has {seeds.shape[0]} entries for {len(per_track)} rows; it "
            "is one predecessor per batch row, not one per segment"
        )

    both = scorer.unary(ev, grid, b, i, j)                       # (S, 2)
    unary = both.gather(-1, lab.unsqueeze(-1)).squeeze(-1)       # (S,)
    prev = _prev_labels(lab, first, b, seeds)
    terms = unary + scorer.transitions()[prev, lab]
    return out.index_add(0, b, terms)


def segmentation_score(
    scorer: SpanScorer, ev: Evidence, segs: Segmentation, grid: Grid
) -> torch.Tensor:
    """Unnormalised score of one track's segmentation -- the batch of one."""
    return segmentation_scores(scorer, ev, [segs], grid)[0]
