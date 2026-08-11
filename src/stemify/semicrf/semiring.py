"""The algebra the segmental recurrence is parameterised over.

``alpha[end][L]`` combines every way of reaching position ``end`` in label ``L``. *How*
they combine is the only thing separating training from decoding: sum the alternatives
and the scan computes ``log Z``; take the best and it computes Viterbi. The recurrence,
the window bookkeeping and the candidate enumeration are identical either way, which is
why they belong to ``scan`` and this belongs here.

A semiring is a set with ⊕ ("add") and ⊗ ("multiply") where ⊗ distributes over ⊕. That
distributivity is not decoration -- it is exactly the step where a shared prefix factors
out of a sum over suffixes, which is what makes the dynamic program valid at all. Swap
in any structure satisfying it and the same recurrence computes a different quantity,
with the proof coming along.

``⊗`` needs no method here: every semiring in this file works in log space, where ⊗ is
ordinary ``+`` and the recurrence adds potentials. A semiring whose elements are not
scalars (entropy carries a pair, k-best carries a list) would need one, and would need
``step_terms`` to broadcast over a trailing element axis -- reachable, deliberately not
built until something asks for it.

**``initial_window`` still hardcodes ``NEG``**, which is sound only because both
semirings below share it as ⊕'s identity. A third whose ``zero`` differs has to thread
this through there too.
"""

from typing import Protocol

import torch

from stemify.semicrf.constants import NEG


class Semiring(Protocol):
    """⊕ for the segmental recurrence, plus its identity.

    Implementations are stateless singletons, not instantiated per call: they carry an
    algebra, not a configuration.
    """

    zero: float
    """⊕'s identity -- the score of *no* path, which any alternative must beat."""

    def reduce(self, terms: torch.Tensor, dim: int) -> torch.Tensor:
        """⊕ over ``dim``. The only operation the forward pass needs."""
        ...

    def reduce_with_witness(
        self, terms: torch.Tensor, dim: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(reduced, index of the surviving element)``, or ``None`` when ⊕ has no
        single survivor to point at.

        Separate from ``reduce`` rather than always returning a pair, because the
        forward pass must not pay for an index it discards -- an argmax is roughly the
        cost of the reduction again, over a scan the length of the track. A caller that
        needs a traceback asks for one; a caller that does not never emits the op.
        """
        ...

    def contract(self, window: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """⊕ over the shared axis of ``window ⊗ trans``: ``(..., K)`` and ``(K, N)`` ->
        ``(..., N)``.

        The semiring matmul, and the reason it exists is memory. The recurrence combines
        an alpha window with a transition matrix over the *previous* label, and doing
        that by broadcasting materialises ``(B, dmax, n_rows, n_labels)`` -- 9 MB per
        step at a 49-label alphabet, ~1.15 GB retained across a chunk. Contracting the
        previous-label axis first never builds it: the result is ``(B, dmax, n_labels)``,
        ~180 KB, and the arithmetic is identical.

        At two labels the saving is nothing and the broadcast would be simpler. At
        forty-nine it is the difference between fitting and not, which is why this is on
        the algebra rather than written out at the one call site.
        """
        ...

    def contract_with_witness(
        self, window: torch.Tensor, trans: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(contracted, index into K of the survivor)``, or ``None`` for a ⊕ with none.

        Viterbi's backpointer now spans *two* reductions -- which previous label won the
        contraction, and which start frame won the reduction after it -- so the traceback
        needs this one as well as ``reduce_with_witness``.
        """
        ...


class _Log:
    """⊕ = ``logsumexp``. Sums over every segmentation, giving ``log Z``.

    The differentiable one, and the reason the objective is trainable: ``logsumexp`` is
    a smooth relaxation of ``max`` (``t·logsumexp(x/t) → max(x)`` as ``t → 0``), so the
    unrolled scan is one autograd graph. No path is discarded, so nothing points at a
    survivor and ``reduce_with_witness`` is ``None`` rather than an error -- "this
    algebra has no argmax" is a fact about the algebra.
    """

    zero = NEG

    def reduce(self, terms: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(terms, dim=dim)

    def reduce_with_witness(
        self, terms: torch.Tensor, dim: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None

    def contract(self, window: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """``logsumexp_k(window[..., k] + trans[k, n])``, as a shifted matmul.

        ``log(exp(w - mw) @ exp(t - mt)) + mw + mt`` -- the log-domain matmul. Both
        operands are shifted to a maximum of 0 before the exponential, so nothing
        overflows and the underflow that remains is correct: a term ``NEG`` below the
        max contributes exactly 0, which is the mass of an unreachable path.

        A row of ``window`` that is *entirely* ``NEG`` (the ramp positions before a track has
        dmax frames behind it) shifts to 0 and comes back at ``NEG + O(log K)``, i.e.
        still unreachable. That is the property ``initial_window`` relies on.
        """
        mw = window.max(dim=-1, keepdim=True).values
        mt = trans.max()
        return torch.log(torch.exp(window - mw) @ torch.exp(trans - mt)) + mw + mt

    def contract_with_witness(
        self, window: torch.Tensor, trans: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return None


class _MaxPlus:
    """⊕ = ``max``. Keeps the single best segmentation -- Viterbi, the tropical semiring.

    ``torch.max`` returns values and indices from one pass, so a traceback costs nothing
    beyond the indices themselves.
    """

    zero = NEG

    def reduce(self, terms: torch.Tensor, dim: int) -> torch.Tensor:
        return terms.max(dim=dim).values

    def reduce_with_witness(
        self, terms: torch.Tensor, dim: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        best = terms.max(dim=dim)
        return best.values, best.indices

    def contract(self, window: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """``max_k(window[..., k] + trans[k, n])``. There is no max-plus matmul primitive, so
        this broadcasts and reduces.

        That rebuilds exactly the tensor ``LOG.contract`` exists to avoid -- and it is
        fine here, because max-plus runs only under ``viterbi``'s ``no_grad``. Nothing
        is retained for a backward pass, so the peak is one step's worth rather than a
        chunk's.
        """
        return (window.unsqueeze(-1) + trans).max(dim=-2).values

    def contract_with_witness(
        self, window: torch.Tensor, trans: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        best = (window.unsqueeze(-1) + trans).max(dim=-2)
        return best.values, best.indices


LOG: Semiring = _Log()
"""Training's algebra: every path, summed."""

MAX_PLUS: Semiring = _MaxPlus()
"""Decoding's algebra: one path, the best.

Which is the narrower claim of the two, and the source of a measured failure -- ``log Z``
accounts for the combinatorially many ways to place ``k`` boundaries while ``max``
compares one segmentation at a time, so the best single path is systematically coarser
than the distribution's own expected note count. See docs/experiments.md.
"""
