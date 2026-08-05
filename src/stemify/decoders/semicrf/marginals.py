"""Span posteriors, and decoding from them instead of from the potentials.

``log Z`` sums over every segmentation; Viterbi maximises over one at a time. There are
combinatorially more ways to be finely segmented than coarsely, so the scorer is fit
knowing each individual fine segmentation must score low to leave the right total mass
on "k notes" -- and ``max`` then pays that per-span cost while never seeing the
multiplicity that justified it. The best *single* path is therefore coarser than the
distribution's own expected note count, and measurably so: on a trained checkpoint the
model expects 0.97 of the reference note count while its Viterbi decode emits 0.38.

A per-segment constant closes some of that (docs/experiments.md), but it is one number
for a quantity whose per-track optimum measured **+0.60 to +2.24** -- and at its own
optimum the decode still over-segments and under-segments simultaneously, which no
single scalar can fix in both directions at once. Marginals are per span. They can add a
boundary where the posterior supports one and drop another in the same track.

**The identity.** For a log-linear model the gradient of the log partition function
with respect to a feature's weight is that feature's expectation. Perturb the score of
one span by ``eps`` and the "feature" is the indicator that the span is present, so

    d log Z / d eps(L, i, j)  =  P(span (L, i, j) is in the segmentation | x)

No backward recurrence is written here. Reverse-mode autodiff through the forward scan
*is* the backward algorithm -- the adjoints it propagates are ``beta`` -- so
``autograd.grad`` evaluates ``alpha ⊗ score ⊗ beta / Z`` on our behalf (Eisner 2016,
*Inside-Outside and Forward-Backward Algorithms Are Just Backprop*). That is why this
module is thirty lines and not a second DP to verify.
"""

import numpy as np
import torch

from stemify.decoders.semicrf.evidence import Evidence
from stemify.decoders.semicrf.forward import log_partitions
from stemify.decoders.semicrf.spans import Grid, SpanScorer

MIN_P = 1e-12
"""Floor under a posterior before its log is taken.

A span the model gives no mass to has ``log P = -inf``, which is the *right* score --
never choose it -- but ``-inf`` propagates into the scan's arithmetic and reaches the
reduction as ``NaN`` rather than as a rejected alternative. ``log(1e-12) ~ -27.6`` is
already far below any span the DP would consider."""


def span_marginals(scorer: SpanScorer, ev: Evidence, grid: Grid) -> torch.Tensor:
    """``(B, T+1, dmax, 2)``: ``P(span | x)`` for every candidate span and label.

    Indexed the way the scan enumerates rather than by ``(i, j)``: entry ``[b, j, d, L]``
    is the span ending at ``j`` that starts at the ``d``-th candidate start, so it lines
    up with ``candidate_starts`` and needs no second copy of that arithmetic.

    Runs under ``enable_grad`` so an ambient ``no_grad`` -- the encoder sets one, and a
    caller may -- cannot silently turn this into a tensor of zeros. Only the scan is
    differentiated: ``inputs=[eps]`` keeps the scorer's own parameters out of it.
    """
    eps = torch.zeros(
        (ev.B, ev.T + 1, grid.dmax, 2),
        dtype=ev.onset_logit.dtype, device=ev.device, requires_grad=True,
    )
    with torch.enable_grad():
        logz = log_partitions(scorer, ev, grid, perturb=eps).sum()
        (marginal,) = torch.autograd.grad(logz, [eps])
    return marginal


class MarginalScorer(SpanScorer):
    """A scorer whose span score is ``log P(span | x)`` under some other scorer.

    Decoding through this is **maximum expected accuracy** (equivalently, minimum Bayes
    risk under a per-span loss): Viterbi still returns a valid tiling, obeying the same
    duration and transition structure it always did, but it now maximises summed
    posterior mass instead of raw potential. The bias this exists to remove lives
    entirely in ``max`` versus ``sum``, so replacing what is maximised removes it without
    touching the model, the training, or the recurrence.

    ``transitions`` is **zero** on purpose. The transition table already shaped the
    distribution these marginals were computed from; scoring it again would count the
    monophonic grammar twice, once inside the posterior and once outside it.

    Not a trained module and never serialised -- ``SemiCRFRecipe`` refuses to write a
    checkpoint for a scorer it cannot rebuild, which is the correct answer for one whose
    "parameters" are a single track's posteriors.
    """

    def __init__(self, marginal: torch.Tensor, grid: Grid, dmax_s: float) -> None:
        super().__init__()
        self.dmax_s = dmax_s
        self._grid = grid
        # A buffer, not a parameter: it is data about one track, and the optimiser must
        # never see it. Registered rather than plain-attributed so `.to(device)` moves it
        # with the module.
        self.register_buffer("log_p", marginal.clamp_min(MIN_P).log())

    def unary(
        self,
        ev: Evidence,
        grid: Grid,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        # The scan hands `i` as candidate *start frames*; the table is indexed by the
        # candidate's offset within the window. `d = i - j + dmax` inverts
        # `candidate_starts`, and the clamp covers the starts that function pins at 0
        # near a track's beginning -- those pair with a `zero` alpha and cannot win, so
        # any finite score there is unreachable rather than wrong.
        table = torch.as_tensor(self.log_p)
        d = (torch.as_tensor(i) - j + grid.dmax).clamp(0, grid.dmax - 1)
        return table[b, j, d]

    def transitions(self) -> torch.Tensor:
        return torch.as_tensor(self.log_p).new_zeros((3, 2))

    def grid(self, frame_times: np.ndarray) -> Grid:
        # The marginals were computed against one grid and are indexed by its `dmax`;
        # deriving a second one from the frame hop could disagree with the table's own
        # shape, so this returns the grid they were built with.
        return self._grid


def marginal_scorer(scorer: SpanScorer, ev: Evidence, grid: Grid) -> MarginalScorer:
    """``MarginalScorer`` over ``scorer``'s own posteriors -- the two-step composed."""
    return MarginalScorer(span_marginals(scorer, ev, grid), grid, scorer.dmax_s)
