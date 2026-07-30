"""Segmental Viterbi: the single best segmentation.

The same recurrence as ``forward``, in the max semiring instead of logsumexp, plus
backpointers to trace the argmax path. Kept separate because the two diverge in more
than the reduction: this one is ``no_grad`` (it is a decode, never an objective),
carries a backpointer table, and ends in a Python traceback loop.
"""

import torch

from stemify.decoders.semicrf.constants import NEG, NOTE, REST, START
from stemify.decoders.semicrf.emissions import Emissions
from stemify.decoders.semicrf.potentials import SemiCRFPotentials
from stemify.decoders.semicrf.window import new_window, push


@torch.no_grad()
def viterbi(
    pot: SemiCRFPotentials, e: Emissions, dmax: int, dur_bucket: torch.Tensor
) -> list[tuple[int, int, int]]:
    """The single highest-scoring segmentation: the same recurrence as the forward
    pass with max in place of logsumexp, plus backpointers to trace the argmax path
    from the terminal state back through START.

    ``int(argf[lab])`` is a device-to-host sync twice per step -- ~14k on a 5-minute
    track -- which is why a CUDA decode measures 4x *slower* than CPU here. See
    ``docs/gpu-throughput-plan.md``; the fix is a tensor backpointer table read out
    once at the end, not a device change."""
    T = e.T
    trans_to = pot.trans_to()  # (3, 2)
    a0 = e.onset_logit.new_full((3,), NEG)
    a0[START] = 0.0
    window, last = new_window(a0, dmax), a0
    back: list[dict | None] = [None]
    for j in range(1, T + 1):
        lo = max(0, j - dmax)
        idx = torch.arange(lo, j, device=e.device)
        note = pot.note_potentials(e, idx, j, dur_bucket)
        rest = pot.rest_potentials(e, idx, j)
        seg = torch.stack([note, rest], dim=-1)          # (W, 2)
        prev = window[dmax - (j - lo):]                  # (W, 3) -- a view
        term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
        flat = term.reshape(-1, 2)                        # (W*3, 2)
        val, argf = flat.max(dim=0)                       # (2,), (2,)
        last = torch.cat([val, val.new_full((1,), NEG)])
        window = push(window, last)
        back.append(
            {lab: (lo + int(argf[lab]) // 3, int(argf[lab]) % 3) for lab in (NOTE, REST)}
        )
    label = int(torch.stack([last[NOTE], last[REST]]).argmax())
    j, segs = T, []
    while j > 0:
        i, prev = back[j][label]
        segs.append((label, i, j))
        j, label = i, prev
        if label == START:
            break
    segs.reverse()
    return segs
