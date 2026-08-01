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
    scorer: SpanScorer, ev: Evidence, grid: Grid
) -> list[tuple[int, int, int]]:
    """The single highest-scoring segmentation of track 0 of ``ev``.

    ``int(argmax[lab])`` is a device-to-host sync twice per step -- ~14k on a 5-minute
    track -- which is why a CUDA decode measures 4x *slower* than CPU here. See
    ``docs/gpu-throughput-plan.md``; the fix is a tensor backpointer table read out once
    at the end, not a device change.
    """
    T = ev.lengths[0]
    if T == 0:
        return []
    window = initial_window(ev.onset_logit, 1, grid.dmax)
    last = window[:, -1]
    back: list[dict | None] = [None]
    for j in range(1, T + 1):
        term = step_terms(scorer, ev, grid, window, j)      # (1, dmax, 3, 2)
        val, arg = term.reshape(-1, 2).max(dim=0)           # (2,), (2,)
        # NEG rather than -inf: START is unreachable after position 0, and a finite
        # sentinel cannot produce a NaN when it meets the potentials added to it.
        last = torch.cat([val, val.new_full((1,), NEG)]).unsqueeze(0)
        window = push(window, last)
        # `term` flattens (dmax, 3=prev), so an argmax index splits as
        # (window offset, prev label) -- and the offset maps back to a frame through
        # the *same* clamped start table the recurrence scored against.
        starts = candidate_starts(grid, j, ev.device)
        back.append(
            {lab: (int(starts[int(arg[lab]) // 3]), int(arg[lab]) % 3)
             for lab in (NOTE, REST)}
        )
    label = int(torch.stack([last[0, NOTE], last[0, REST]]).argmax())
    j, segs = T, []
    while j > 0:
        i, prev = back[j][label]
        segs.append((label, i, j))
        j, label = i, prev
        if label == START:
            break
    segs.reverse()
    return segs
