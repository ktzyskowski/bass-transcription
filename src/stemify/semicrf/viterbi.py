"""Segmental Viterbi: the single best segmentation.

The *recurrence* is ``scan`` under the max-plus semiring -- not a second loop. What is
left here is the half no semiring can supply: a semiring reduces alternatives to a
score, and a score is not a path. Turning backpointers back into spans is Viterbi's own
job, and it is the only thing in this file.

That traceback stays a sequential Python loop because it is inherently one -- each
step's segment start decides where the next lookup lands -- but it runs once per
*segment*, not per frame, over CPU-resident data.
"""

import torch

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.forward import candidate_starts, scan
from stemify.semicrf.semiring import MAX_PLUS
from stemify.semicrf.spans import Grid, SpanScorer
from stemify.semicrf.window import initial_window


@torch.no_grad()
def viterbi(
    scorer: SpanScorer, ev: Evidence, grid: Grid, note_bias: float = 0.0
) -> list[tuple[int, int, int]]:
    """The single highest-scoring segmentation of track 0 of ``ev``.

    ``note_bias`` is added once per ``note`` span, so a segmentation with ``k`` notes
    is shifted by ``k * note_bias``. It exists because this reduction and the objective
    that trained the scorer disagree about what they are computing. Training minimises
    ``log Z − score(gold)``, and ``log Z`` sums over *every* segmentation -- so it
    counts the combinatorially many ways to place ``k`` boundaries, and the scorer is
    fit knowing each individual fine segmentation must therefore score low to leave the
    right total mass on "``k`` notes". ``max`` sees one segmentation at a time. It feels
    the per-span cost that multiplicity paid for and never sees the multiplicity, so the
    best *single* path is systematically coarser than the distribution's own expected
    note count -- the same bias ASR corrects with a word insertion penalty.

    Zero by default: the correction belongs to whoever tuned it against a validation
    set, and a decode silently applying someone else's constant is worse than one that
    does not correct at all.

    ``no_grad`` because this is a decode, never an objective. Learning happens through
    the *forward* semiring, which is differentiable; ``max`` has no useful gradient.

    Backpointers accumulate as **one device tensor**, read back after the scan rather
    than element by element inside it: converting an argmax to a Python ``int`` mid-loop
    is a device-to-host sync, and a sync drains the queue. A ``(T+1, B, 2)`` int64 table
    buys one transfer in place of O(T) of them.
    """
    T = ev.lengths[0]
    if T == 0:
        return []
    # Every *pitch* label collects the bias, rest collects none -- the correction is
    # per emitted note, and under a wide alphabet "a note" is any of the pitch labels
    # rather than the integer 0. At two labels this is exactly `bias[NOTE]`.
    bias = ev.onset_logit.new_zeros(scorer.labels.n_labels)
    bias[: scorer.labels.n_pitches] = note_bias
    back = ev.onset_logit.new_zeros(
        (T + 1, ev.B, scorer.labels.n_labels), dtype=torch.long
    )
    window = scan(
        scorer, ev, grid, initial_window(ev.onset_logit, ev.B, grid.dmax, scorer.labels.n_rows),
        stop=T, semiring=MAX_PLUS, bias=bias, witness=back,
    )

    last = window[:, -1]
    label = int(last[0, : scorer.labels.n_labels].argmax())
    cpu_back = back[:, 0].cpu()                             # the one transfer
    cpu = torch.device("cpu")
    j, segs = T, []
    while j > 0:
        # `term` flattened (dmax, 3=prev), so an argmax index splits as
        # (window offset, previous label) -- and the offset maps back to a frame
        # through the *same* clamped start table the recurrence scored against,
        # rather than through a second copy of that arithmetic.
        flat = int(cpu_back[j, label])
        n_rows = scorer.labels.n_rows
        i = int(candidate_starts(grid, j, cpu)[flat // n_rows])
        prev = flat % n_rows
        segs.append((label, i, j))
        j, label = i, prev
        if label == scorer.labels.START:
            break
    segs.reverse()
    return segs
