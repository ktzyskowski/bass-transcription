"""The algebra the segmental recurrence is parameterised over.

``alpha[j][L]`` combines every way of reaching position ``j`` in label ``L``. *How*
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


LOG: Semiring = _Log()
"""Training's algebra: every path, summed."""

MAX_PLUS: Semiring = _MaxPlus()
"""Decoding's algebra: one path, the best.

Which is the narrower claim of the two, and the source of a measured failure -- ``log Z``
accounts for the combinatorially many ways to place ``k`` boundaries while ``max``
compares one segmentation at a time, so the best single path is systematically coarser
than the distribution's own expected note count. See docs/experiments.md.
"""
