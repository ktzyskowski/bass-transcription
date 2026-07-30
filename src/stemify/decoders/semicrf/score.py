"""Score of *one given* segmentation under the potentials.

The third way to score a tiling, alongside ``forward`` (logsumexp over all of them)
and ``viterbi`` (the best one). The NLL is ``log_partition - score(gold)``, so this is
the gold half -- but nothing here knows the segmentation is gold, only that it is a
list of ``(label, i, j)`` spans.

Separate from ``gold.py`` on purpose: that module decides *which* segmentation is gold
(the audible-gap rule, pure numpy) and changes when the labelling rule changes; this
one turns any segmentation into a number and changes when the *potentials* change.
Different reasons to edit, so different files.
"""

import torch

from stemify.decoders.semicrf.constants import NOTE, START
from stemify.decoders.semicrf.emissions import BatchEmissions, Emissions
from stemify.decoders.semicrf.potentials import SemiCRFPotentials

# One segment: (label, start frame, end frame), end exclusive.
Segmentation = list[tuple[int, int, int]]


def _terms(
    pot: SemiCRFPotentials,
    onset: torch.Tensor,
    Vc: tuple[torch.Tensor, torch.Tensor],
    Uc: tuple[torch.Tensor, torch.Tensor],
    Jc: tuple[torch.Tensor, torch.Tensor],
    lab: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    prev: torch.Tensor,
    dur_bucket: torch.Tensor,
) -> torch.Tensor:
    """Per-segment ``unary + incoming transition`` for a flat set of segments.

    Each cue arrives already gathered as a ``(start, end)`` pair of ``(S,)`` tensors,
    so one body serves both the per-track and the batched score -- the only difference
    between them is how the caller gathered.

    Both label branches are evaluated for every segment and selected with ``where``.
    That is ~2x the arithmetic and a fraction of the cost of branching: the loop this
    replaced launched a handful of kernels *per segment*, measured at 7-70x the graphed
    forward step itself. Every index it touches is in range for either label --
    ``i + 1 <= T`` because ``i < T``, and ``j - i`` is in ``[1, dmax]`` -- so the unused
    branch cannot fault."""
    note = (
        pot.w_on * onset
        + pot.w_v * (Vc[1] - Vc[0])
        + pot.w_pitch * (Jc[1] - Jc[0])  # interior changes only; caller offsets by 1
        + pot.b_note
        + pot.dur[dur_bucket[j - i]]
    )
    rest = pot.w_u * (Uc[1] - Uc[0]) + pot.b_rest
    unary = torch.where(lab == NOTE, note, rest)
    return unary + pot.trans_to()[prev, lab]


def _prev_labels(lab: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """The label preceding each segment, ``START`` wherever ``first`` is set."""
    prev = torch.cat([lab.new_full((1,), START), lab[:-1]])
    return torch.where(first, torch.full_like(prev, START), prev)


def segmentation_score(
    pot: SemiCRFPotentials,
    e: Emissions,
    segs: Segmentation,
    dur_bucket: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised score of one track's segmentation: the sum of its segments' unary
    potentials and the transitions between them."""
    if not segs:
        return e.onset_logit.new_zeros(())
    g = torch.tensor(segs, dtype=torch.long, device=e.device)  # (S, 3)
    lab, i, j = g[:, 0], g[:, 1], g[:, 2]
    first = torch.zeros_like(lab, dtype=torch.bool)
    first[0] = True
    return _terms(
        pot, e.onset_logit[i],
        (e.Vc[i], e.Vc[j]), (e.Uc[i], e.Uc[j]), (e.Jc[i + 1], e.Jc[j]),
        lab, i, j, _prev_labels(lab, first), dur_bucket,
    ).sum()


def segmentation_scores(
    pot: SemiCRFPotentials,
    batch: BatchEmissions,
    per_track: list[Segmentation],
    dur_bucket: torch.Tensor,
) -> torch.Tensor:
    """``(B,)`` scores for a padded minibatch, in one vectorised pass.

    Calling ``segmentation_score`` per track in a Python loop was the single largest
    cost in a training step -- a batch's worth of segments is thousands of tiny kernel
    launches. Flattening every track's segments into one indexed set makes the whole
    batch a fixed handful of ops, and ``index_add`` folds them back per track.

    Indices are never clamped, so a caller passing a span past a track's true length
    would silently read the flat padding ``pad_emissions`` leaves there."""
    rows = [(b, *seg) for b, segs in enumerate(per_track) for seg in segs]
    out = batch.onset_logit.new_zeros(len(per_track))
    if not rows:
        return out
    t = torch.tensor(rows, dtype=torch.long, device=batch.device)  # (S, 4)
    b, lab, i, j = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    # A track's first segment follows START; `rows` is grouped by track, so a change
    # in `b` marks each boundary.
    first = torch.ones_like(b, dtype=torch.bool)
    first[1:] = b[1:] != b[:-1]
    terms = _terms(
        pot, batch.onset_logit[b, i],
        (batch.Vc[b, i], batch.Vc[b, j]),
        (batch.Uc[b, i], batch.Uc[b, j]),
        (batch.Jc[b, i + 1], batch.Jc[b, j]),
        lab, i, j, _prev_labels(lab, first), dur_bucket,
    )
    return out.index_add(0, b, terms)
