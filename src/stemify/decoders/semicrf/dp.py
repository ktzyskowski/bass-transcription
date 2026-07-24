"""The segmental dynamic program: one recurrence, three uses.

``alpha[j][label]`` is the score of the best (Viterbi) or total (forward) way to tile
``[0, j)`` ending in a ``label`` span. The recurrence is the same in all three; only
the semiring changes:

  - ``log_partition`` / ``batched_log_partition`` -- **logsumexp** (sum semiring):
    ``log Z``, differentiable, for the NLL.
  - ``viterbi`` -- **max** (with backpointers): the single best segmentation, for
    ``decode``.

Free functions over ``(potentials, emissions)`` -- they never touch a raw parameter,
so each is testable against a hand-set ``SemiCRFPotentials``. See ``docs/semi-crf.md``.
"""

import torch

from stemify.decoders.semicrf.constants import NEG, NOTE, REST, START
from stemify.decoders.semicrf.emissions import BatchEmissions, Emissions
from stemify.decoders.semicrf.potentials import SemiCRFPotentials


def log_partition(
    pot: SemiCRFPotentials, e: Emissions, dmax: int, dur_bucket: torch.Tensor
) -> torch.Tensor:
    """``log Z``: the logsumexp over every segmentation, via the segmental forward
    recurrence. ``alpha[j][label]`` is the mass of all ways to tile ``[0, j)`` ending
    in a ``label`` span; differentiable, so ``d log Z`` gives the model's expected
    segment features."""
    T = e.T
    trans_to = pot.trans_to()  # (3, 2)
    a0 = e.onset_logit.new_full((3,), NEG)
    a0[START] = 0.0
    alpha = [a0]
    for j in range(1, T + 1):
        lo = max(0, j - dmax)
        idx = torch.arange(lo, j)
        note = pot.note_potentials(e, idx, j, dur_bucket)
        rest = pot.rest_potentials(e, idx, j)
        seg = torch.stack([note, rest], dim=-1)          # (W, 2=next)
        prev = torch.stack(alpha[lo:j], dim=0)           # (W, 3=prev)
        term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
        val = torch.logsumexp(term.reshape(-1, 2), dim=0)  # (2,)
        alpha.append(torch.cat([val, val.new_full((1,), NEG)]))
    return torch.logsumexp(alpha[T][:2], dim=0)


def batched_log_partition(
    pot: SemiCRFPotentials, batch: BatchEmissions, dmax: int, dur_bucket: torch.Tensor
) -> torch.Tensor:
    """``log Z`` for every track in a padded minibatch: the same recurrence as
    ``log_partition`` run over ``(B, ...)`` tensors in one ``j``-loop, read out at
    each track's true length. Equal to the per-track scan track-for-track; this is
    only the throughput form (one kernel per step, not per track)."""
    onset, Vc, Uc, Jc = batch.onset_logit, batch.Vc, batch.Uc, batch.Jc
    B, Tmax = onset.shape
    trans_to = pot.trans_to()                        # (3, 2)
    neg_col = onset.new_full((B, 1), NEG)

    a0 = onset.new_full((B, 3), NEG)
    a0[:, START] = 0.0
    alpha = [a0]                                      # position 0: must be START
    for j in range(1, Tmax + 1):
        lo = max(0, j - dmax)
        idx = torch.arange(lo, j)
        d = j - idx                                  # (W,)
        vsum = Vc[:, j : j + 1] - Vc[:, idx]         # (B, W)
        usum = Uc[:, j : j + 1] - Uc[:, idx]
        jmp = Jc[:, j : j + 1] - Jc[:, idx + 1]      # interior pitch changes
        dur_term = pot.dur[dur_bucket[d]]            # (W,)
        note = (pot.w_on * onset[:, idx] + pot.w_v * vsum + pot.w_pitch * jmp
                + pot.b_note + dur_term.unsqueeze(0))
        rest = pot.w_u * usum + pot.b_rest
        seg = torch.stack([note, rest], dim=-1)      # (B, W, 2=next)
        prev = torch.stack(alpha[lo:j], dim=1)       # (B, W, 3=prev)
        term = prev.unsqueeze(-1) + trans_to.view(1, 1, 3, 2) + seg.unsqueeze(2)
        val = torch.logsumexp(term.reshape(B, -1, 2), dim=1)  # (B, 2)
        alpha.append(torch.cat([val, neg_col], dim=1))        # (B, 3)
    final = torch.stack([alpha[batch.lengths[b]][b] for b in range(B)])  # (B, 3)
    return torch.logsumexp(final[:, :2], dim=1)      # (B,)


def gold_score(
    pot: SemiCRFPotentials,
    e: Emissions,
    gold: list[tuple[int, int, int]],
    dur_bucket: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised score of the gold segmentation: the sum of its segments' unary
    potentials and the transitions between them."""
    score = e.onset_logit.new_zeros(())
    prev = START
    for label, i, j in gold:
        idx = torch.tensor([i])
        if label == NOTE:
            unary = pot.note_potentials(e, idx, j, dur_bucket)[0]
        else:
            unary = pot.rest_potentials(e, idx, j)[0]
        trans = pot.init[label] if prev == START else pot.trans[prev, label]
        score = score + unary + trans
        prev = label
    return score


@torch.no_grad()
def viterbi(
    pot: SemiCRFPotentials, e: Emissions, dmax: int, dur_bucket: torch.Tensor
) -> list[tuple[int, int, int]]:
    """The single highest-scoring segmentation: the same recurrence as the forward
    pass with max in place of logsumexp, plus backpointers to trace the argmax path
    from the terminal state back through START."""
    T = e.T
    trans_to = pot.trans_to()  # (3, 2)
    a0 = e.onset_logit.new_full((3,), NEG)
    a0[START] = 0.0
    alpha = [a0]
    back: list[dict | None] = [None]
    for j in range(1, T + 1):
        lo = max(0, j - dmax)
        idx = torch.arange(lo, j)
        note = pot.note_potentials(e, idx, j, dur_bucket)
        rest = pot.rest_potentials(e, idx, j)
        seg = torch.stack([note, rest], dim=-1)          # (W, 2)
        prev = torch.stack(alpha[lo:j], dim=0)           # (W, 3)
        term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
        flat = term.reshape(-1, 2)                        # (W*3, 2)
        val, argf = flat.max(dim=0)                       # (2,), (2,)
        alpha.append(torch.cat([val, val.new_full((1,), NEG)]))
        back.append(
            {lab: (lo + int(argf[lab]) // 3, int(argf[lab]) % 3) for lab in (NOTE, REST)}
        )
    label = int(torch.stack([alpha[T][NOTE], alpha[T][REST]]).argmax())
    j, segs = T, []
    while j > 0:
        i, prev = back[j][label]
        segs.append((label, i, j))
        j, label = i, prev
        if label == START:
            break
    segs.reverse()
    return segs
