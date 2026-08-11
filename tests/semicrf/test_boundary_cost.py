"""``Δ(y, gold)``: the boundary cost the structured margin adds inside the partition.

The property everything else rests on is ``Δ(gold, gold) = 0`` -- without it, ``margin``
at 0 would not reproduce the plain NLL and no sweep would have a control.
"""

import torch

from stemify.semicrf.cost import boundary_cost

T, DMAX = 14, 10
GOLD = [(0, 0, 4), (1, 4, 6), (0, 6, 10), (1, 10, 14)]   # boundaries at 0, 4, 6, 10


def _cost(table: torch.Tensor, i: int, j: int, row: int = 0) -> float:
    d = i - j + DMAX
    assert 0 <= d < DMAX, f"[{i},{j}) is longer than dmax={DMAX}"
    return float(table[row, j, d, 0])


def _table(golds=(GOLD,), tol: int = 1) -> torch.Tensor:
    return boundary_cost(list(golds), T, DMAX, torch.device("cpu"), tolerance_frames=tol)


def test_gold_costs_nothing():
    """``margin`` at 0 must be today's objective exactly, which needs this."""
    table = _table()
    assert [_cost(table, i, j) for _, i, j in GOLD] == [0.0] * len(GOLD)


def test_a_merged_span_pays_per_swallowed_boundary():
    table = _table()
    assert _cost(table, 0, 6) == 1.0        # swallows 4
    assert _cost(table, 4, 10) == 1.0       # swallows 6
    assert _cost(table, 0, 10) == 2.0       # swallows 4 and 6


def test_a_split_pays_for_a_boundary_gold_does_not_have():
    """Priced as well as merges, so a margin cannot buy recall with fragmentation."""
    assert _cost(_table(), 2, 4) == 1.0


def test_sub_tolerance_jitter_is_free():
    """The note metric matches within 50 ms -- one frame at the DP hop -- so a boundary
    one frame out is *correct*, and charging it would price an error the thing we
    measure does not consider one. It is also exactly the slop a plateaued onset target
    leaves the model indifferent about, so the cost would be fighting jitter."""
    table = _table(tol=1)
    assert _cost(table, 3, 6) == 0.0        # starts 1 early against gold's 4
    assert _cost(table, 5, 6) == 0.0        # starts 1 late against gold's 4
    assert _cost(table, 7, 10) == 0.0       # starts 1 late against gold's 6
    # Two frames out is past tolerance -- and this one also merges across 4, so it
    # pays for both errors independently.
    assert _cost(table, 2, 6) == 2.0


def test_a_zero_tolerance_cost_charges_that_jitter():
    """The tolerance is doing work, not decoration: the same span, forgiven at the
    metric's tolerance and charged without it."""
    assert _cost(_table(tol=1), 5, 6) == 0.0
    assert _cost(_table(tol=0), 5, 6) == 1.0


def test_spans_before_gold_s_first_boundary_are_free():
    """Under chunked training the window's gold holds segments *ending* in the scanned
    region, so boundaries deeper in the left context belong to the previous chunk and
    are not in view. Charging a span for starting there prices it against evidence we
    were not given; under-penalising is the safe direction."""
    late = [(0, 6, 10), (1, 10, 14)]                    # nothing known below frame 6
    table = _table(golds=(late,))
    assert _cost(table, 2, 6) == 0.0
    # Inside the known region and past tolerance of gold's 6, so charged.
    assert _cost(table, 8, 10) == 1.0


def test_an_empty_gold_charges_nothing():
    table = _table(golds=([],))
    assert torch.count_nonzero(table) == 0


def test_rows_are_scored_against_their_own_gold():
    """One table, one gold per batch row. A row indexed against its neighbour's gold is
    a wrong cost that changes no shape."""
    other = [(0, 0, 6), (1, 6, 14)]                     # boundaries at 0, 6
    table = _table(golds=(GOLD, other))
    assert _cost(table, 4, 6, row=0) == 0.0             # gold for row 0
    assert _cost(table, 4, 6, row=1) == 1.0             # not a boundary for row 1
