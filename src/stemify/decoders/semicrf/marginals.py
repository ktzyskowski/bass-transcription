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

from collections.abc import Sequence

import numpy as np
import torch

from stemify.decoders.semicrf.evidence import Evidence
from stemify.decoders.semicrf.forward import log_partitions
from stemify.decoders.semicrf.spans import Grid, SpanScorer

HALF = 0.5
"""The posterior a span must beat to be worth including.

Not a tuned threshold -- it is what the decision rule implies. Minimising the expected
symmetric difference between the emitted segmentation and the truth,

    E[L(y, y')] = |y| + E|y'| - 2 * sum over s in y of P(s)

is maximising ``sum over s in y of (P(s) - 1/2)``. A span pays a half to exist and earns
its posterior back, so it is included exactly when the model thinks it more likely
present than absent. Change this and you are changing how the loss weighs a false
positive against a false negative, which is a statement about the metric rather than a
knob."""


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
    """A scorer whose span score is ``P(span | x) - 1/2`` under some other scorer.

    **Linear in the posterior, not logarithmic**, and the distinction is the whole
    rule. Summing ``log P`` would score a segmentation by its probability, so every
    additional span multiplies in another factor below one -- reintroducing the exact
    per-span cost that makes MAP under-segment, on top of marginals computed to remove
    it. Measured, that decoded *coarser* than MAP: 0.44 of the reference note count
    against MAP-plus-bias's 0.98. Summing ``P - 1/2`` scores a segmentation by its
    expected agreement with the truth, which is what the metric asks for and what has no
    length preference of its own.

    Decoding through this is **maximum expected accuracy** (equivalently, minimum Bayes
    risk under a per-span loss): Viterbi still returns a valid tiling, obeying the same
    duration and transition structure it always did, but it now maximises summed
    posterior mass instead of raw potential. The bias this exists to remove lives
    entirely in ``max`` versus ``sum``, so replacing what is maximised removes it without
    touching the model, the training, or the recurrence.

    ``transitions`` is **zero** on purpose. The transition table already shaped the
    distribution these marginals were computed from; scoring it again would count the
    monophonic grammar twice, once inside the posterior and once outside it. The
    structure it imposes survives anyway -- Viterbi still returns a valid tiling, so
    spans cannot overlap or leave gaps whatever the posteriors say.

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
        self.register_buffer("gain", marginal - HALF)

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
        table = torch.as_tensor(self.gain)
        d = (torch.as_tensor(i) - j + grid.dmax).clamp(0, grid.dmax - 1)
        return table[b, j, d]

    def transitions(self) -> torch.Tensor:
        return torch.as_tensor(self.gain).new_zeros((3, 2))

    def grid(self, frame_times: np.ndarray) -> Grid:
        # The marginals were computed against one grid and are indexed by its `dmax`;
        # deriving a second one from the frame hop could disagree with the table's own
        # shape, so this returns the grid they were built with.
        return self._grid


def boundary_marginals(span: torch.Tensor) -> torch.Tensor:
    """``(B, T+1, 2)``: ``P(a label-L span ends at j | x)``, from the span posteriors.

    Summing over candidate starts, which is the whole point. A segmentation contains at
    most one span ending at ``j``, so the sum is a probability rather than an
    expectation -- and it aggregates the mass that ``span_marginals`` leaves scattered
    across neighbouring ``(i, j)`` pairs when the model is sure a note *starts around
    here* and unsure exactly where.

    That scattering is not hypothetical. On a trained checkpoint, 63 exact spans clear
    a posterior of 0.5 against **253 boundaries** -- so a rule that decides span by span
    sees a model that is unsure about everything, and the same rule one level up sees
    one that is sure about four times as much.
    """
    return span.sum(dim=2)


def tolerated(boundary: torch.Tensor, frames: int) -> torch.Tensor:
    """Boundary posteriors summed over ``+-frames``: the mass the *matcher* would accept.

    Every decision rule tried before this one scored the posterior of an **exact**
    position -- an exact span, then an exact boundary frame. The note metric never asks
    for that. It matches within ``ONSET_TOLERANCE_S`` (50 ms), which at the DP's 34.8 ms
    hop is about +-1.4 frames, so a boundary one frame early still counts and its exact
    position's posterior does not.

    That is why all three collapsed: they demanded the model be certain about something
    nothing scores it on, found it certain almost nowhere, and fell back to the fewest
    spans. Summing over the window asks the question the metric asks -- *is there a note
    boundary near here* -- which is also the question the model is decisive about.

    A sum rather than a max because these are disjoint events: a segmentation has at
    most one boundary at each frame, so mass at neighbouring frames adds.
    """
    if frames <= 0:
        return boundary
    width = 2 * frames + 1
    # (B, T+1, 2) -> (B, 2, T+1) for a 1-D pool over time, and back.
    pooled = torch.nn.functional.avg_pool1d(
        boundary.transpose(1, 2), kernel_size=width, stride=1, padding=frames,
        count_include_pad=False,
    ) * width
    return pooled.transpose(1, 2)


class BoundaryScorer(SpanScorer):
    """Scores a span by the posterior that a boundary falls at its **end**.

    Each boundary is counted once: a segmentation's score is the sum over its own
    boundary positions, because the span ending at ``j`` is the only one that can claim
    ``j``. Scoring both endpoints would double-count every interior boundary and, worse,
    weigh the two ends of the track differently from its middle.

    Independent of ``i``, which is exactly the point. ``MarginalScorer`` asks "is this
    the right span", a question the model answers diffusely because it is unsure of the
    exact start; this asks "is there a note boundary here", which it answers sharply.
    The duration prior is not lost -- Viterbi still returns a valid tiling, so span
    lengths remain constrained by ``dmax`` and the transition structure.
    """

    def __init__(self, boundary: torch.Tensor, grid: Grid, dmax_s: float,
                 tau: float = HALF) -> None:
        super().__init__()
        self.dmax_s = dmax_s
        self._grid = grid
        self.register_buffer("gain", boundary - tau)

    def unary(
        self,
        ev: Evidence,
        grid: Grid,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        table = torch.as_tensor(self.gain)
        value = table[b, j]
        # Broadcast across the candidate-start axis without indexing on it: the score
        # genuinely does not depend on `i`, and adding a zero of `i`'s shape is what
        # lets one implementation serve both the scan (b=(B,1), i=(1,dmax)) and the gold
        # scorer (three (S,) tensors), as `SpanScorer.unary` requires.
        spread = torch.zeros_like(torch.as_tensor(i), dtype=value.dtype)
        return value + spread.unsqueeze(-1)

    def transitions(self) -> torch.Tensor:
        return torch.as_tensor(self.gain).new_zeros((3, 2))

    def grid(self, frame_times: np.ndarray) -> Grid:
        return self._grid


def marginal_scorer(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    aggregate: str = "boundary",
    tau: float = HALF,
    tol_frames: int = 0,
) -> SpanScorer:
    """A posterior-scoring ``SpanScorer`` over ``scorer``'s own distribution.

    ``aggregate="span"`` scores each exact span, ``"boundary"`` scores each boundary
    position. They are the same posteriors summed at different levels, and the level
    decides whether the rule works: measured on a trained checkpoint, ``"span"`` decodes
    to 0.31 of the reference note count -- *coarser* than the MAP decode it corrects --
    because almost no exact span clears ``tau`` and the rule then minimises span count.
    """
    span = span_marginals(scorer, ev, grid)
    if aggregate == "span":
        return MarginalScorer(span, grid, scorer.dmax_s)
    if aggregate == "boundary":
        return BoundaryScorer(
            tolerated(boundary_marginals(span), tol_frames), grid, scorer.dmax_s, tau
        )
    raise ValueError(f"aggregate must be 'span' or 'boundary', got {aggregate!r}")


def segment_confidence(
    marginal: torch.Tensor,
    segs: Sequence[tuple[int, int, int]],
    grid: Grid,
    row: int = 0,
) -> dict[tuple[int, int, int], float]:
    """``P(span | x)`` for each decoded segment, keyed by ``(label, i, j)``.

    What a reviewer needs and a MAP decode does not produce. Viterbi returns the best
    path and says nothing about how close the runner-up was, so every note arrives
    equally certain -- which on a real recording means checking all of them or none.
    The posterior is already computed for the marginals; this is the lookup.

    **Read it as a ranking, not as P(this note is correct).** It is the posterior of
    this *exact* span -- both endpoints, to the frame. The note metric matches within
    +-50 ms, so a note whose boundary is one frame off still counts, while its exact
    span's posterior does not. Measured on a trained checkpoint the values run
    min 0.03 / median 0.28 / max 0.80: the mass for "a note around here" is spread
    across neighbouring ``(i, j)`` pairs, which is the same fragmentation that defeats
    decoding from these directly. So 0.28 is a floor on the note being right, not an
    estimate of it, and the number worth acting on is which notes sit at the bottom.
    """
    return {
        (label, i, j): float(marginal[row, j, min(max(i - j + grid.dmax, 0), grid.dmax - 1), label])
        for label, i, j in segs
    }
