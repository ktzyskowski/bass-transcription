"""Segmental Viterbi: the single best segmentation.

The same recurrence as ``forward`` -- literally the same ``step_terms`` -- reduced with
``max`` instead of ``logsumexp``, plus backpointers to trace the argmax path.

The *reduction* stays separate rather than being folded into a shared semiring, because
the two differ in cost rather than in logic. This one is ``no_grad`` (it is a decode,
never an objective), keeps a backpointer table, reads an argmax back to the host every
step, and ends in a Python traceback loop. Routing the forward pass through a common
reducer would make it pay for an argmax it discards and give up the static shapes its
CUDA-graph capture needs.
"""

import torch

from stemify.decoders.semicrf.constants import NEG, NOTE, REST, START
from stemify.decoders.semicrf.evidence import Evidence
from stemify.decoders.semicrf.forward import candidate_starts, step_terms
from stemify.decoders.semicrf.spans import Grid, SpanScorer
from stemify.decoders.semicrf.window import initial_window, push


@torch.no_grad()
def viterbi(
    scorer: SpanScorer, ev: Evidence, grid: Grid, note_bias: float = 0.0
) -> list[tuple[int, int, int]]:
    """The single highest-scoring segmentation of track 0 of ``ev``.

    ``note_bias`` is added once per ``note`` span, so a segmentation with ``k`` notes
    is shifted by ``k * note_bias``. It exists because this reduction and the objective
    that trained the scorer disagree about what they are computing. Training minimises
    ``log Z - score(gold)``, and ``log Z`` sums over *every* segmentation -- so it
    counts the combinatorially many ways to place ``k`` boundaries, and the scorer is
    fit knowing each individual fine segmentation must therefore score low to leave the
    right total mass on "``k`` notes". ``max`` sees one segmentation at a time. It feels
    the per-span cost that multiplicity paid for and never sees the multiplicity, so the
    best *single* path is systematically coarser than the distribution's own expected
    note count -- the same bias ASR corrects with a word insertion penalty.

    Zero by default: the correction belongs to whoever tuned it against a validation
    set, and a decode silently applying someone else's constant is worse than one that
    does not correct at all.

    The backpointers accumulate as **one device tensor**, read back after the scan
    rather than element by element inside it: converting an argmax to a Python ``int``
    mid-loop is a device-to-host sync, and a sync drains the queue. A ``(T+1, 2)``
    int64 table buys one transfer in place of O(T) of them, which is the whole reason
    it is not a list of dicts.

    The traceback stays a sequential Python loop because it is inherently one -- each
    step's segment start decides where the next lookup lands -- but it runs once per
    *segment*, not per frame, over CPU-resident data.
    """
    T = ev.lengths[0]
    if T == 0:
        return []
    window = initial_window(ev.onset_logit, 1, grid.dmax)
    last = window[:, -1]
    back = ev.onset_logit.new_zeros((T + 1, 2), dtype=torch.long)
    # Added to the *next-label* axis, so it lands once per span closed as a note and
    # never on a rest. Built once, outside the scan.
    bias = ev.onset_logit.new_zeros(2)
    bias[NOTE] = note_bias
    for j in range(1, T + 1):
        term = step_terms(scorer, ev, grid, window, j)      # (1, dmax, 3, 2)
        val, arg = (term + bias).reshape(-1, 2).max(dim=0)  # (2,), (2,)
        back[j] = arg
        # NEG rather than -inf: START is unreachable after position 0, and a finite
        # sentinel cannot produce a NaN when it meets the potentials added to it.
        last = torch.cat([val, val.new_full((1,), NEG)]).unsqueeze(0)
        window = push(window, last)

    label = int(torch.stack([last[0, NOTE], last[0, REST]]).argmax())
    back = back.cpu()                                       # the one transfer
    cpu = torch.device("cpu")
    j, segs = T, []
    while j > 0:
        # `term` flattened (dmax, 3=prev), so an argmax index splits as
        # (window offset, previous label) -- and the offset maps back to a frame
        # through the *same* clamped start table the recurrence scored against,
        # rather than through a second copy of that arithmetic.
        # `flat`, not `arg`: the forward loop above already binds `arg` to the batched
        # argmax *tensor*, and reusing the name here for one scalar backpointer made
        # two different things read as one.
        flat = int(back[j, label])
        i = int(candidate_starts(grid, j, cpu)[flat // 3])
        prev = flat % 3
        segs.append((label, i, j))
        j, label = i, prev
        if label == START:
            break
    segs.reverse()
    return segs
