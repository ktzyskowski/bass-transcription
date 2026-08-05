"""The algebra ``scan`` is parameterised over, tested without the dynamic program.

``scan`` used to hardcode ``logsumexp`` and ``viterbi`` carried a second copy of the
loop to hardcode ``max``. Merging them puts one line between training's answer and
decoding's, so what has to hold is that the two semirings really are different algebras
over the same reduction -- and that the seam refuses the one combination that would fail
silently.
"""

import pytest
import torch

from stemify.decoders.semicrf.semiring import LOG, MAX_PLUS


@pytest.fixture
def terms():
    torch.manual_seed(0)
    return torch.randn(2, 12, 2)          # (B, alternatives, label)


# --------------------------------------------------------------------------- #
# Each semiring is the algebra it claims                                       #
# --------------------------------------------------------------------------- #
def test_log_sums_the_alternatives(terms):
    assert torch.allclose(LOG.reduce(terms, dim=1), torch.logsumexp(terms, dim=1))


def test_max_plus_keeps_the_best(terms):
    assert torch.allclose(MAX_PLUS.reduce(terms, dim=1), terms.max(dim=1).values)


def test_log_dominates_max_plus(terms):
    """`logsumexp >= max` always, with equality only in the limit of one alternative
    carrying all the mass. This is the whole relationship between the two answers: the
    forward pass is a soft Viterbi, and its gap over the hard one is the mass held by
    every path that is not the best -- exactly the mass a MAP decode discards."""
    assert (LOG.reduce(terms, dim=1) >= MAX_PLUS.reduce(terms, dim=1)).all()


def test_the_gap_closes_as_one_alternative_takes_over(terms):
    """Scaling the scores up is annealing: `t·logsumexp(x/t) → max(x)` as `t → 0`. Pins
    that the two are the same recurrence at different temperatures rather than two
    unrelated reductions."""
    gaps = [
        float((LOG.reduce(terms * s, dim=1) - MAX_PLUS.reduce(terms * s, dim=1)).mean())
        for s in (1.0, 10.0, 100.0)
    ]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert gaps[-1] < 1e-3


# --------------------------------------------------------------------------- #
# The witness contract                                                         #
# --------------------------------------------------------------------------- #
def test_max_plus_witness_points_at_the_value_it_returned(terms):
    """A backpointer that does not index the winning alternative reconstructs a
    different segmentation than the one that was scored -- and the score would still
    look right, because the score came from the reduction, not the index."""
    value, index = MAX_PLUS.reduce_with_witness(terms, dim=1)
    assert torch.allclose(value, terms.gather(1, index.unsqueeze(1)).squeeze(1))


def test_max_plus_witness_agrees_with_its_own_plain_reduce(terms):
    """Two entry points to one operation; a caller that asks for backpointers must not
    thereby get a different answer."""
    value, _ = MAX_PLUS.reduce_with_witness(terms, dim=1)
    assert torch.allclose(value, MAX_PLUS.reduce(terms, dim=1))


def test_log_reports_that_it_has_no_witness(terms):
    """`None`, not an exception and not an arbitrary index. A sum keeps every
    alternative, so there is no survivor to point at -- that is a fact about the
    algebra, and the caller decides whether it is a problem."""
    assert LOG.reduce_with_witness(terms, dim=1) is None


def test_both_semirings_share_an_identity_for_now(terms):
    """`initial_window` still hardcodes `NEG` as the unreachable score, which is only
    sound while every semiring agrees on it. If this ever fails, that hardcoding is a
    live bug rather than a documented shortcut."""
    assert LOG.zero == MAX_PLUS.zero
