"""Span posteriors: ``P(span | x)`` for every candidate span, from one autograd call.

**The identity.** For a log-linear model the gradient of the log partition function with
respect to a feature's weight is that feature's expectation. Perturb the score of one
span by ``eps`` and the "feature" is the indicator that the span is present, so

    d log Z / d eps(L, start, end)  =  P(span (L, start, end) is in the segmentation | x)

No backward recurrence is written here. Reverse-mode autodiff through the forward scan
*is* the backward algorithm -- the adjoints it propagates are ``beta`` -- so
``autograd.grad`` evaluates ``alpha ⊗ score ⊗ beta / Z`` on our behalf (Eisner 2016,
*Inside-Outside and Forward-Backward Algorithms Are Just Backprop*). That is why this
module is thirty lines and not a second DP to verify.

What the posteriors are *for* is reporting, not deciding: the per-note confidence
``decode`` attaches, and the labels-free expected note count that diagnosed the Viterbi
under-segmentation. Four decision rules that decoded from them instead -- scoring exact
spans, exact boundaries, and boundaries within the matcher's tolerance -- each lost to a
single tuned constant, and are recorded in ``docs/experiments.md`` rather than here.
"""

from collections.abc import Sequence

import torch

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.forward import log_partitions
from stemify.semicrf.spans import SpanScorer


def span_marginals(scorer: SpanScorer, evidence: Evidence, dmax: int) -> torch.Tensor:
    """``(B, T+1, dmax, n_labels)``: ``P(span | x)`` for every candidate span and label.

    Indexed the way the scan enumerates rather than by ``(start, end)``: entry ``[row, end, d, L]``
    is the span ending at ``end`` that starts at the ``d``-th candidate start, so it lines
    up with ``candidate_starts`` and needs no second copy of that arithmetic.

    Runs under ``enable_grad`` so an ambient ``no_grad`` -- the encoder sets one, and a
    caller may -- cannot silently turn this into a tensor of zeros. Only the scan is
    differentiated: ``inputs=[eps]`` keeps the scorer's own parameters out of it.
    """
    # Width from the scorer's own alphabet. This was a literal `2`, which was right
    # for as long as every scorer was {note, rest} -- and under a pitch alphabet it
    # raises on the first `term + perturb[:, end]`, taking the confidence decode and
    # the expected-note-count diagnostic with it.
    eps = torch.zeros(
        (evidence.n_tracks, evidence.n_frames + 1, dmax, scorer.labels.n_labels),
        dtype=evidence.onset_logit.dtype, device=evidence.device, requires_grad=True,
    )
    with torch.enable_grad():
        logz = log_partitions(scorer, evidence, dmax, perturb=eps).sum()
        (marginal,) = torch.autograd.grad(logz, [eps])
    return marginal


def segment_confidence(
    marginal: torch.Tensor,
    segs: Sequence[tuple[int, int, int]],
    dmax: int,
    row: int = 0,
) -> dict[tuple[int, int, int], float]:
    """``P(span | x)`` for each decoded segment, keyed by ``(label, start, end)``.

    What a reviewer needs and a MAP decode does not produce. Viterbi returns the best
    path and says nothing about how close the runner-up was, so every note arrives
    equally certain -- which on a real recording means checking all of them or none.
    The posterior is already computed for the marginals; this is the lookup.

    **Read it as a ranking, not as P(this note is correct).** It is the posterior of
    this *exact* span -- both endpoints, to the frame. The note metric matches within
    +-50 ms, so a note whose boundary is one frame off still counts, while its exact
    span's posterior does not. Measured on a trained checkpoint the values run
    min 0.03 / median 0.28 / max 0.80: the mass for "a note around here" is spread
    across neighbouring ``(start, end)`` pairs, which is the same fragmentation that defeats
    decoding from these directly. So 0.28 is a floor on the note being right, not an
    estimate of it, and the number worth acting on is which notes sit at the bottom.
    """
    return {
        (label, start, end): float(marginal[row, end, min(max(start - end + dmax, 0), dmax - 1), label])
        for label, start, end in segs
    }
