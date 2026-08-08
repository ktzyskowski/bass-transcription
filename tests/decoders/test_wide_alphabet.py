"""The recurrence at a label alphabet wider than two, against brute-force enumeration.

`test_semicrf.py` checks the DP against explicit enumeration, which is the only test in
the suite that cannot be passed by two copies of the same wrong recurrence -- but it does
so at two labels, where the previous-label contraction is a 3x2 matrix and an off-by-one
in it has almost nowhere to hide.

These run the same construction at **five** labels, which is where the generalisation
actually has to hold: the contraction is a 6x5, the packed backpointer's radix is 6
rather than 3, and `arg_prev` must be gathered at the winning *start* rather than reduced
on its own. Every one of those is a change that leaves shapes valid and answers wrong.

The scorer is a table of random potentials rather than `LinearSpanScorer`: what is under
test is the recurrence's handling of an alphabet, not any particular parameterisation,
and a table makes every (span, label) score independent so nothing can accidentally
agree.
"""

import itertools

import numpy as np
import pytest
import torch
from torch import nn

from stemify.core.contour import Contour
from stemify.semicrf.evidence import Evidence
from stemify.semicrf.forward import log_partitions
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.score import segmentation_score
from stemify.semicrf.spans import Grid, SpanScorer
from stemify.semicrf.viterbi import viterbi

T, DMAX, DT = 6, 3, 0.032
LABELS = LabelSet(lo=40, hi=43)          # four pitches + rest = five labels


class _TableScorer(SpanScorer):
    """Independent random potentials per ``(i, j, label)`` and per transition.

    Deliberately structureless. A scorer with real features could mask an indexing bug
    by scoring neighbouring labels similarly; independent draws mean any mis-indexing
    lands on an unrelated number and the totals diverge.
    """

    def __init__(self, labels: LabelSet, seed: int = 0) -> None:
        super().__init__()
        self.labels = labels
        self.dmax_s = DMAX * DT
        g = torch.Generator().manual_seed(seed)
        self.u = nn.Parameter(torch.randn(T + 1, T + 1, labels.n_labels, generator=g))
        self.t = nn.Parameter(torch.randn(labels.n_rows, labels.n_labels, generator=g))

    def unary(self, ev, grid, b, i, j):          # type: ignore[no-untyped-def]
        return self.u[i, j]

    def transitions(self) -> torch.Tensor:
        return self.t

    def grid(self, frame_times: np.ndarray) -> Grid:
        return Grid(DMAX, torch.zeros(DMAX + 1, dtype=torch.long))


def _evidence() -> Evidence:
    ft = np.arange(T) * DT
    zeros = torch.zeros(1, T + 1)
    return Evidence(
        onset_logit=torch.zeros(1, T), voiced_c=zeros, rest_c=zeros,
        lengths=[T], frame_times=[ft],
        contours=[Contour(times=ft, freqs=np.full(T, 110.0, dtype=np.float32))],
    )


def _all_segmentations(labels: LabelSet):
    """Every tiling of ``[0, T)`` into spans of length 1..DMAX, over the full alphabet."""
    def rec(pos):
        if pos == T:
            yield []
            return
        for d in range(1, min(DMAX, T - pos) + 1):
            for lab in range(labels.n_labels):
                for tail in rec(pos + d):
                    yield [(lab, pos, pos + d), *tail]
    yield from rec(0)


@pytest.fixture(scope="module")
def setup():
    scorer = _TableScorer(LABELS)
    ev = _evidence()
    grid = scorer.grid(ev.frame_times[0])
    segs = list(_all_segmentations(LABELS))
    scored = torch.stack([segmentation_score(scorer, ev, s, grid) for s in segs])
    return scorer, ev, grid, segs, scored


def test_the_enumeration_is_big_enough_to_be_a_test(setup):
    _, _, _, segs, _ = setup
    # Five labels over tilings of six frames: if this collapses to a handful, the
    # oracle below is not exercising the contraction at all.
    assert len(segs) > 2000, f"only {len(segs)} segmentations"


def test_log_partition_equals_brute_force_at_five_labels(setup):
    scorer, ev, grid, _, scored = setup
    got = log_partitions(scorer, ev, grid)[0]
    brute = torch.logsumexp(scored, dim=0)
    assert torch.allclose(got, brute, atol=1e-4), (
        f"log Z {got.item()} != brute force {brute.item()}"
    )


def test_viterbi_returns_the_best_segmentation_at_five_labels(setup):
    scorer, ev, grid, segs, scored = setup
    best = segs[int(scored.argmax())]
    got = viterbi(scorer, ev, grid)
    # The path, not just its score. A backpointer whose radix or gather is wrong still
    # produces a *scoring* path -- it is the identity of the segmentation that catches it.
    assert got == best, f"viterbi returned {got}, enumeration says {best}"
    assert torch.allclose(
        segmentation_score(scorer, ev, got, grid), scored.max(), atol=1e-4
    )


def test_every_label_is_reachable_by_the_traceback():
    """A packed backpointer with the wrong radix decodes to a *valid* label, so an
    alphabet where only the low labels ever win would hide it. Bias each label in turn
    and check the decode follows -- if the radix were 3 rather than n_rows, the high
    labels would be unreachable and this would fail on them alone."""
    ev, seen = _evidence(), set()
    for label in range(LABELS.n_labels):
        scorer = _TableScorer(LABELS, seed=7)
        with torch.no_grad():
            scorer.u[:, :, label] += 20.0        # dominate every span at this label
        got = viterbi(scorer, ev, scorer.grid(ev.frame_times[0]))
        seen.update(lab for lab, _, _ in got)
    assert seen == set(range(LABELS.n_labels)), (
        f"labels {sorted(set(range(LABELS.n_labels)) - seen)} were never decoded"
    )


def test_the_alphabet_width_reaches_every_shape(setup):
    scorer, ev, grid, _, _ = setup
    # The shapes the width has to flow through, pinned so a future change that hardcodes
    # 2 or 3 again fails here rather than in a training run.
    assert scorer.transitions().shape == (LABELS.n_rows, LABELS.n_labels) == (6, 5)
    assert scorer.unary(ev, grid, 0, torch.arange(DMAX), T).shape[-1] == LABELS.n_labels
    assert len(set(itertools.chain.from_iterable(
        [(lab,) for lab, _, _ in viterbi(scorer, ev, grid)]
    ))) >= 1
