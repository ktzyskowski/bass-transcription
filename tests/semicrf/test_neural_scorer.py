"""``NeuralSpanScorer`` against brute-force enumeration, and its table against itself.

The recurrence is already checked against enumeration with a table of independent
potentials (``test_wide_alphabet``). What is new here is *how the potentials arrive*:
this scorer does not compute a span's score when asked for it, it precomputes every
span into a ``(B, T+1, dmax, n_labels)`` tensor and answers by gathering. That moves the
risk from the arithmetic to the indexing, and an indexing bug there is quiet -- every
shape still matches, every score is still some real span's score, and the DP happily
returns the best segmentation under a permuted potential.

Three access patterns read that one table and must agree: the scan's ``(B, 1) x
(1, dmax)`` dmax of candidates, the gold scorer's scattered ``(S,)`` triples, and a
single span addressed by three ints. Enumeration pins all of them at once, because the
brute-force score goes through the gold path while ``log Z`` goes through the scan.
"""

import math

import numpy as np
import pytest
import torch

from stemify.transcription.frames import Frames
from stemify.semicrf import neural
from stemify.semicrf.evidence import build_evidence
from stemify.semicrf.forward import log_partitions
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.neural import NeuralSpanScorer
from stemify.semicrf.score import segmentation_score
from stemify.semicrf.viterbi import viterbi

T, DMAX, DT, TRUNK = 6, 3, 0.032, 5
BINS = 12                                  # 4 semitones at 3 bins each
LABELS = LabelSet.from_bins(BINS)          # 4 pitches + rest = 5 labels


def _evidence(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ft = np.arange(T) * DT
    frames = Frames(
        heads={
            "onset": torch.rand(T, generator=g),
            "voicing": torch.rand(T, generator=g),
            "pitch": torch.randn(T, BINS, generator=g),
        },
        frame_times=ft,
        hidden=torch.randn(T, TRUNK, generator=g),
    )
    return build_evidence([frames], stride=1)


def _scorer(seed: int = 0) -> NeuralSpanScorer:
    torch.manual_seed(seed)
    sc = NeuralSpanScorer(
        LABELS, trunk_dim=TRUNK, frame_dim=4, order=2, span_dim=6, dur_dim=3,
        dmax_s=DMAX * DT, dmax_frames=DMAX,
    )
    # Off the zero-init readout. With `label_emb` at zero every label scores alike bar
    # its pitch term, which is a fine place to *start training* and a poor place to test
    # an indexing bug from -- a permuted table would still agree with enumeration.
    with torch.no_grad():
        for p in sc.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    return sc


def _all_segmentations():
    """Every tiling of ``[0, T)`` into spans of length 1..DMAX, over the full alphabet."""
    def rec(pos):
        if pos == T:
            yield []
            return
        for d in range(1, min(DMAX, T - pos) + 1):
            for lab in range(LABELS.n_labels):
                for tail in rec(pos + d):
                    yield [(lab, pos, pos + d), *tail]
    yield from rec(0)


@pytest.fixture(scope="module")
def setup():
    scorer, ev = _scorer(), _evidence()
    dmax = scorer.dmax_for(ev.frame_times[0])
    segs = list(_all_segmentations())
    scored = torch.stack([segmentation_score(scorer, ev, s, dmax) for s in segs])
    return scorer, ev, dmax, segs, scored


def test_log_partition_equals_brute_force(setup):
    scorer, ev, dmax, _, scored = setup
    got = log_partitions(scorer, ev, dmax)[0]
    assert torch.allclose(got, torch.logsumexp(scored, dim=0), atol=1e-4), (
        f"log Z {got.item()} != brute force {torch.logsumexp(scored, 0).item()}"
    )


def test_viterbi_returns_the_enumerated_best(setup):
    scorer, ev, dmax, segs, scored = setup
    got = viterbi(scorer, ev, dmax)
    assert got == segs[int(scored.argmax())]


def test_the_scan_and_the_gold_scorer_read_the_same_table(setup):
    """The two access patterns, on identical spans.

    The scan asks with ``b=(B,1)``, ``i=(1,dmax)`` and an int ``j``; the gold scorer asks
    with three ``(S,)`` tensors. Both index ``span_scores[b, j, j - i - 1]``, and a
    transposed or off-by-one gather is only visible where the two disagree.
    """
    scorer, ev, dmax, _, _ = setup
    j = 5
    starts = torch.arange(j - DMAX, j).clamp(min=0)
    grid_style = scorer.unary(ev, dmax, torch.zeros(1, 1, dtype=torch.long),
                              starts.unsqueeze(0), j)                    # (1, dmax, L)
    flat_style = scorer.unary(ev, dmax, torch.zeros(DMAX, dtype=torch.long),
                              starts, torch.full((DMAX,), j))            # (dmax, L)
    assert torch.allclose(grid_style[0], flat_style, atol=1e-6)
    for k, i in enumerate(starts.tolist()):
        one = scorer.unary(ev, dmax, 0, i, j)                            # (L,)
        assert torch.allclose(one, flat_style[k], atol=1e-6), f"start {i}"


def test_slicing_the_precompute_does_not_change_it(monkeypatch, setup):
    """``J_BLOCK`` bounds memory, not the answer. Spans are independent, so the blocks
    concatenate -- unless a block's local ``j`` leaks in where the global one belongs,
    which is exactly the bug this shape invites."""
    scorer, ev, dmax, _, _ = setup
    whole = scorer.precompute(ev, dmax)
    monkeypatch.setattr(neural, "J_BLOCK", 2)
    assert torch.allclose(whole, scorer.precompute(ev, dmax), atol=1e-6)


def test_every_parameter_gets_gradient():
    """A zero-initialised readout can sever the whole network below it: with both
    ``label_emb`` and the gate at zero, ``d score / d f`` is identically zero and
    ``phi``, ``f`` and ``dur`` train on nothing. Caught once already."""
    scorer, ev = NeuralSpanScorer(
        LABELS, trunk_dim=TRUNK, frame_dim=4, span_dim=6, dur_dim=3,
        dmax_s=DMAX * DT, dmax_frames=DMAX,
    ), _evidence()
    log_partitions(scorer, ev, scorer.dmax_for(ev.frame_times[0])).sum().backward()
    dead = [n for n, p in scorer.named_parameters()
            if p.grad is None or not torch.any(p.grad != 0)]
    assert not dead, f"no gradient reaches {dead}"


def test_transitions_depend_on_the_interval_not_the_pitch():
    """The factorisation's whole claim: a fifth is a fifth wherever it starts.

    A full ``(n_rows, n_labels)`` table would pass everything else here, so this is what
    says the interval embedding is actually wired to the interval."""
    scorer = _scorer()
    trans = scorer.transitions()
    n = LABELS.n_pitches
    for p in range(n - 1):
        for q in range(n - 1):
            assert torch.allclose(trans[p, q], trans[p + 1, q + 1], atol=1e-6), (
                f"({p}->{q}) and ({p + 1}->{q + 1}) are the same interval"
            )
    # ...and rest is *not* on that ladder: it is a class, not a pitch.
    assert not torch.allclose(trans[0, LABELS.REST], trans[1, 0], atol=1e-6)


def test_rest_starts_as_good_as_an_uninformative_pitch():
    scorer = NeuralSpanScorer(LABELS, trunk_dim=TRUNK, dmax_s=DMAX * DT, dmax_frames=DMAX)
    assert scorer.rest_logprob.item() == pytest.approx(-math.log(LABELS.n_pitches))


def test_a_trunkless_evidence_is_refused():
    """``Frames.hidden`` is optional, so the scorer that needs it must say so -- and it
    must say so *here*, not by pooling zeros into a plausible number."""
    ft = np.arange(T) * DT
    frames = Frames(
        heads={"onset": torch.rand(T), "voicing": torch.rand(T),
               "pitch": torch.randn(T, BINS)},
        frame_times=ft,
    )
    ev = build_evidence([frames], stride=1)
    scorer = _scorer()
    with pytest.raises(ValueError, match="carries none"):
        log_partitions(scorer, ev, scorer.dmax_for(ev.frame_times[0]))


def test_a_hop_the_duration_table_was_not_fitted_for_is_refused():
    scorer = _scorer()
    with pytest.raises(ValueError, match="duration embedding is sized"):
        scorer.dmax_for(np.arange(T) * (DT / 2))


def test_span_marginals_match_brute_force(setup):
    """``P(span | x)`` from autograd against the enumerated posterior.

    ``span_marginals`` sized its perturbation tensor with a literal 2, which was right
    for exactly as long as every scorer was ``{note, rest}``. Under a pitch alphabet it
    raises -- taking the confidence decode and the expected-note-count diagnostic with
    it -- so this pins the width to the scorer's own alphabet.
    """
    from stemify.semicrf.marginals import span_marginals

    scorer, ev, dmax, segs, scored = setup
    posterior = torch.softmax(scored, dim=0)
    brute = torch.zeros(1, T + 1, DMAX, LABELS.n_labels)
    for p, seg in zip(posterior, segs, strict=True):
        for label, i, j in seg:
            brute[0, j, i - j + DMAX, label] += p
    got = span_marginals(scorer, ev, dmax)
    assert got.shape == brute.shape
    assert torch.allclose(got, brute, atol=1e-4), (
        f"max |diff| {(got - brute).abs().max().item()}"
    )


def test_expected_note_count_equals_the_enumerated_expectation(setup):
    """The diagnostic that separates "the readout under-produces" from "the model does".

    ``E[count] = sum over spans of P(span)`` by linearity, so it is a sum over the
    marginals' pitch labels -- no scorer parameter named, which is what lets it read a
    neural scorer as well as the linear one it was written against.
    """
    from stemify.semicrf.marginals import span_marginals

    scorer, ev, dmax, segs, scored = setup
    posterior = torch.softmax(scored, dim=0)
    brute = sum(
        float(p) * sum(1 for lab, _, _ in seg if LABELS.is_note(lab))
        for p, seg in zip(posterior, segs, strict=True)
    )
    got = float(span_marginals(scorer, ev, dmax)[..., : LABELS.n_pitches].sum())
    assert got == pytest.approx(brute, abs=1e-3), f"{got} != {brute}"
