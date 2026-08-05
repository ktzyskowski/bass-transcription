"""SemiCRFDecoder -- gold construction, a differentiable NLL that trains, and a
Viterbi decode that recovers the notes after a few fit steps. No audio, no model:
synthetic onset/voicing curves and a pyin-style contour drive the whole path.

The last section sweeps the same paths over ``cpu`` and ``cuda`` (see the ``device``
fixture) and asserts the two agree numerically. Running is not enough: a device fix
that quietly changes the arithmetic would still "work" everywhere, so parity against
CPU is what actually pins the behaviour.
"""

import numpy as np
import pytest
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.decoders.semicrf.constants import NOTE, REST, START
from stemify.decoders.semicrf.forward import (
    chunk_log_partition,
    log_partitions,
    scan,
)
from stemify.decoders.semicrf.marginals import (
    MarginalScorer,
    marginal_scorer,
    segment_confidence,
    span_marginals,
)
from stemify.decoders.semicrf.semiring import MAX_PLUS
from stemify.decoders.semicrf.spans import Grid
from stemify.decoders.semicrf.window import initial_window
from stemify.decoders.semicrf.viterbi import viterbi
from stemify.decoders.semicrf.score import segmentation_score, segmentation_scores
from stemify.decoders.semicrf.evidence import Evidence, build_evidence
from stemify.decoders.semicrf.gold import gold_segments
from stemify.notes import NoteEvent

SPANS = [(0.2, 0.9), (1.2, 1.9)]
HZ = [110.0, 146.0]
MIDI = [45, 50]

DT = 0.0116  # feature hop, seconds
T = 240
TIMES = np.arange(T) * DT  # ~2.8 s grid


def _curve(spans: list[tuple[float, float]], hi: float, lo: float) -> np.ndarray:
    """A per-frame curve at ``hi`` inside any ``(start, end)`` span, else ``lo``."""
    y = np.full(T, lo, dtype=np.float32)
    for a, b in spans:
        y[(TIMES >= a) & (TIMES < b)] = hi
    return y


def _two_note_cues(
    note_spans: list[tuple[float, float]], pitches_hz: list[float]
) -> Cues:
    """Cues for held notes: an onset spike at each start, voicing high inside the
    notes, and a contour holding each note's pitch (unvoiced between)."""
    onset = np.full(T, 0.01, dtype=np.float32)
    for a, _ in note_spans:
        onset[round(a / DT)] = 0.99
    voicing = _curve(note_spans, 0.99, 0.01)
    freqs = np.zeros(T, dtype=np.float32)
    for (a, b), hz in zip(note_spans, pitches_hz):
        freqs[(TIMES >= a) & (TIMES < b)] = hz
    return Cues(
        contour=Contour(times=TIMES, freqs=freqs),
        onset=onset, frame_times=TIMES, voicing=voicing,
    )


def _notes(spans: list[tuple[float, float]], midis: list[float]) -> list[NoteEvent]:
    return [NoteEvent(onset=a, offset=b, midi_pitch=m) for (a, b), m in zip(spans, midis)]


def test_gold_rest_across_audible_gap():
    # 0.9 -> 1.2 is a 0.3 s gap (> GAP_REST_S): a real rest between the two notes.
    dec = SemiCRFDecoder()
    e = build_evidence(_two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0]), dec.stride)
    dmax = dec.scorer.grid(e.frame_times[0]).dmax
    gold = gold_segments(e.frame_times[0], _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)
    labels = [lab for lab, _, _ in gold]
    assert labels.count(NOTE) == 2
    assert REST in labels  # the audible gap became a rest


def test_gold_closes_a_note_off_artifact_gap():
    # 0.5 -> 0.55 is a 0.05 s gap (< GAP_REST_S): a note-off artifact, so the notes
    # are made contiguous (note -> note) with no interior rest.
    dec = SemiCRFDecoder()
    e = build_evidence(_two_note_cues([(0.2, 0.5), (0.55, 0.9)], [110.0, 110.0]), dec.stride)
    dmax = dec.scorer.grid(e.frame_times[0]).dmax
    gold = gold_segments(e.frame_times[0], _notes([(0.2, 0.5), (0.55, 0.9)], [45, 45]), dmax)
    # No rest between the notes: everything from the first note to the last is NOTE.
    note_idx = [k for k, (lab, _, _) in enumerate(gold) if lab == NOTE]
    between = gold[note_idx[0] : note_idx[-1] + 1]
    assert all(lab == NOTE for lab, _, _ in between)


def _score_by_loop(scorer, ev, gold, grid):
    """The pre-vectorisation reference: one segment at a time, straight from the
    definition ``sum of unary + incoming transition``.

    It shares ``scorer.unary`` with the code under test, deliberately. What this pins
    is the *enumeration and transition bookkeeping* -- which segments exist, what
    precedes each -- and that is where a recurrence goes subtly wrong. The unary
    formula now has exactly one definition by construction, so pinning it against a
    hand-copy would only assert that a copy was made correctly."""
    score = ev.onset_logit.new_zeros(())
    prev = START
    for label, i, j in gold:
        score = score + scorer.unary(ev, grid, 0, i, j)[label]
        score = score + scorer.transitions()[prev, label]
        prev = label
    return score


def _fitted_scorer(dec):
    """Move the potentials off their symmetric init, so a bug that happens to cancel
    at w=1/bias=0 cannot pass."""
    with torch.no_grad():
        dec.scorer.w_on.fill_(0.7)
        dec.scorer.w_v.fill_(1.3)
        dec.scorer.w_u.fill_(0.4)
        dec.scorer.b_note.fill_(0.25)
        dec.scorer.b_rest.fill_(-0.4)
        dec.scorer.dur.copy_(torch.linspace(-1.0, 1.0, dec.scorer.dur.numel()))
        dec.scorer.trans.copy_(torch.tensor([[0.3, -0.2], [-0.5, 0.1]]))
        dec.scorer.init.copy_(torch.tensor([0.6, -0.3]))
    return dec


def test_gold_score_matches_the_per_segment_loop():
    # The vectorised gold_score evaluates both label branches and selects with
    # `where`; the loop it replaced branched. They must agree exactly.
    dec = _fitted_scorer(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_evidence(cues, dec.stride)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    gold = gold_segments(e.frame_times[0], _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)
    assert len(gold) > 2  # the test is vacuous if gold is trivial

    fast = segmentation_score(dec.scorer, e, gold, grid)
    ref = _score_by_loop(dec.scorer, e, gold, grid)
    assert torch.allclose(fast, ref, atol=1e-5), f"{fast.item()} != {ref.item()}"


def test_gold_score_gradient_matches_the_per_segment_loop():
    # gold_score is half the NLL, so its gradient is what trains the potentials.
    dec = _fitted_scorer(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_evidence(cues, dec.stride)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    gold = gold_segments(e.frame_times[0], _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)

    g_fast = torch.autograd.grad(
        segmentation_score(dec.scorer, e, gold, grid), list(dec.scorer.parameters()),
        retain_graph=True, allow_unused=True,
    )
    g_ref = torch.autograd.grad(
        _score_by_loop(dec.scorer, e, gold, grid), list(dec.scorer.parameters()),
        allow_unused=True,
    )
    for a, b in zip(g_fast, g_ref, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.allclose(a, b, atol=1e-5)


def test_gold_scores_batched_equals_per_track():
    # The batched form flattens every track's segments into one indexed set and folds
    # them back with index_add; a mis-grouped `first` mask would mis-assign the START
    # transition, which only shows up with more than one track.
    dec = _fitted_scorer(SemiCRFDecoder())
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    na = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    ns = _notes([(0.2, 0.9)], [45])

    ev = build_evidence([a, short], dec.stride)
    grid = dec.scorer.grid(ev.frame_times[0])
    golds = [
        gold_segments(ft, n, grid.dmax)
        for ft, n in zip(ev.frame_times, [na, ns], strict=True)
    ]

    batched = segmentation_scores(dec.scorer, ev, golds, grid)
    # ``one(b)`` re-cuts row b to its own length, so the single-track path never sees
    # the flat padding the batch carries past the short track's end.
    single = torch.stack([
        segmentation_score(dec.scorer, ev.one(b), g, grid)
        for b, g in enumerate(golds)
    ])
    assert torch.allclose(batched, single, atol=1e-4)


def test_gold_scores_handles_a_track_with_no_segments():
    # An empty note list yields empty gold; index_add must leave that row at zero
    # rather than dropping the row and shifting every later track's score.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9)], [110.0])
    e = build_evidence(cues, dec.stride)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    gold = gold_segments(e.frame_times[0], _notes([(0.2, 0.9)], [45]), dmax)

    scores = segmentation_scores(dec.scorer, build_evidence([cues, cues], dec.stride), [[], gold], grid)
    assert scores[0].item() == 0.0
    assert torch.allclose(scores[1], segmentation_score(dec.scorer, e, gold, grid), atol=1e-4)


def _truncate(ev, n):
    """The first ``n`` DP frames of an Evidence, so chunks tile it exactly."""
    return Evidence(
        ev.onset_logit[:, :n], ev.voiced_c[:, : n + 1], ev.rest_c[:, : n + 1],
        [n], [ev.frame_times[0][:n]], ev.contours,
    )


def _long_emissions(dec, seconds=25.0, seed=0):
    """A synthetic track long enough that ``dmax`` (115 DP frames, 4 s) fits several
    times over -- the module-level fixture is only 80 DP frames, shorter than one
    ``dmax``, so it cannot exercise a carry at all."""
    n = round(seconds / DT)
    times = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    onset = rng.uniform(0.01, 0.99, n).astype(np.float32)
    voicing = rng.uniform(0.01, 0.99, n).astype(np.float32)
    freqs = np.where(voicing > 0.5, 110.0, 0.0).astype(np.float32)
    cues = Cues(
        contour=Contour(times=times, freqs=freqs),
        onset=onset, frame_times=times, voicing=voicing,
    )
    return build_evidence(cues, dec.stride)


def _stack_evidence(es):
    """Concatenate equal-length single-track ``Evidence`` into one batch.

    The library builds a batch from *cues*; these tests build evidence directly from
    random tensors to drive the recurrence without a contour, so they need their own
    way to batch it. Equal lengths only -- padding is what ``build_evidence`` is for."""
    return Evidence(
        torch.cat([e.onset_logit for e in es]),
        torch.cat([e.voiced_c for e in es]),
        torch.cat([e.rest_c for e in es]),
        [e.lengths[0] for e in es],
        [e.frame_times[0] for e in es],
        [e.contours[0] for e in es],
    )


def _all_segmentations(T, dmax):
    """Every way to tile ``[0, T)`` into labelled spans of length 1..dmax.

    Ground truth by construction -- no recurrence, no dynamic programming, just
    enumeration. This is what the DP is *supposed* to sum (forward) and maximise
    (Viterbi) over."""
    def rec(pos):
        if pos == T:
            yield []
            return
        for d in range(1, min(dmax, T - pos) + 1):
            for lab in (NOTE, REST):
                for tail in rec(pos + d):
                    yield [(lab, pos, pos + d), *tail]
    yield from rec(0)


def _random_emissions(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    ft = np.arange(T) * DT
    contour = Contour(times=ft, freqs=np.full(T, 110.0, dtype=np.float32))
    cs = lambda: torch.cumsum(torch.randn(T + 1, generator=g), 0).unsqueeze(0)
    return Evidence(
        torch.randn(1, T, generator=g), cs(), cs(), [T], [ft], [contour],
    )


def test_log_partition_equals_brute_force_enumeration():
    # THE correctness test for the forward recurrence. Every other DP test in this
    # file compares one of our implementations against another -- batched vs single,
    # chunked vs whole, cpu vs cuda -- which two copies of the *same* wrong recurrence
    # would pass. This one enumerates all segmentations explicitly and sums them.
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    segs = list(_all_segmentations(T, dmax))
    assert len(segs) > 100, f"only {len(segs)} segmentations -- not a real test"
    brute = torch.logsumexp(
        torch.stack([_score_by_loop(dec.scorer, e, s, grid) for s in segs]), dim=0
    )
    got = log_partitions(dec.scorer, e, grid)[0]
    assert torch.allclose(got, brute, atol=1e-4), (
        f"log Z {got.item()} != brute-force {brute.item()} over {len(segs)} segmentations"
    )


def test_viterbi_equals_brute_force_argmax():
    # Same ground truth, max semiring: the returned segmentation must be the single
    # highest-scoring one out of every possible tiling.
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T, seed=1)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    scored = [
        (_score_by_loop(dec.scorer, e, s, grid).item(), s)
        for s in _all_segmentations(T, dmax)
    ]
    best_score, best = max(scored, key=lambda kv: kv[0])
    got = viterbi(dec.scorer, e, grid)
    got_score = _score_by_loop(dec.scorer, e, got, grid).item()
    assert abs(got_score - best_score) < 1e-4, (
        f"viterbi scored {got_score}, best possible {best_score}"
    )
    assert got == best, f"viterbi returned {got}, brute force says {best}"


def test_scan_under_max_plus_equals_the_best_score_by_enumeration():
    """The merged loop, checked at the *scan* level rather than through the traceback.

    `test_viterbi_equals_brute_force_argmax` above exercises the same path but can only
    fail once backpointers are also wrong; this pins the reduction alone. Ground truth
    is enumeration, not our other recurrence -- one loop now serves both semirings, so a
    mistake in it would move `log Z` and Viterbi together and stay invisible to any test
    that compares them to each other."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T, seed=1)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    best = max(
        _score_by_loop(dec.scorer, e, s, grid).item()
        for s in _all_segmentations(T, dmax)
    )
    window = scan(
        dec.scorer, e, grid, initial_window(e.onset_logit, 1, dmax),
        stop=T, semiring=MAX_PLUS,
    )
    got = float(window[0, -1, :2].max().detach())
    assert abs(got - best) < 1e-4, f"max-plus scan says {got}, enumeration says {best}"


# --------------------------------------------------------------------------- #
# Span marginals                                                               #
# --------------------------------------------------------------------------- #
def _enumerated_marginals(scorer, ev, grid, T, dmax):
    """`P(span in y)` by explicit enumeration: every segmentation, its probability,
    summed over the ones containing each span. Ground truth by construction."""
    segs = list(_all_segmentations(T, dmax))
    scores = torch.tensor([_score_by_loop(scorer, ev, s, grid).item() for s in segs])
    probs = (scores - torch.logsumexp(scores, 0)).exp()
    want = torch.zeros(T + 1, dmax, 2)
    for p, seg in zip(probs, segs):
        for label, i, j in seg:
            want[j, i - j + dmax, label] += p
    return want


def test_span_marginals_equal_brute_force_enumeration():
    """THE correctness test for the marginals, and it cannot be replaced by an internal
    one. `d log Z / d eps` is only the posterior if the scan it differentiates is the
    right recurrence *and* the perturbation lands on the right axis -- a shape mistake
    that shares eps across the wrong dimension still yields a plausible, normalised,
    entirely wrong table. Enumeration has no recurrence to be wrong about."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T, seed=5)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    want = _enumerated_marginals(dec.scorer, e, grid, T, dmax)
    got = span_marginals(dec.scorer, e, grid)[0]
    assert torch.allclose(got, want, atol=1e-5), (
        f"max abs error {float((got - want).abs().max())}"
    )


def test_something_must_end_at_the_last_frame():
    """A segmentation tiles `[0, T)`, so the spans ending at `T` are mutually exclusive
    and exhaustive: their posteriors sum to exactly 1. Cheap, and it catches a table
    that is right in shape but off by a normaliser."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 7, 3
    e = _random_emissions(T, seed=6)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))
    m = span_marginals(dec.scorer, e, grid)[0]
    assert float(m[T].sum()) == pytest.approx(1.0, abs=1e-5)


def test_marginals_and_the_note_count_gradient_agree():
    """Two routes to the same expectation: summing `P(note span)` over every span, and
    differentiating `log Z` by `b_note` -- whose feature *is* the note count. They come
    from the same identity applied at different granularity, so disagreement means the
    perturbation is not scoring the event its axis claims."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T, seed=7)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    by_marginal = float(span_marginals(dec.scorer, e, grid)[0, :, :, NOTE].sum())
    logz = log_partitions(dec.scorer, e, grid).sum()
    (by_gradient,) = torch.autograd.grad(logz, [dec.scorer.b_note])
    assert by_marginal == pytest.approx(float(by_gradient), abs=1e-4)


def test_the_span_score_is_linear_in_the_posterior_not_logarithmic():
    """The bug this exists for, and the validity test below does not catch it.

    Scoring a segmentation by `sum of log P` scores it by its *probability*, so every
    extra span multiplies in another factor below one -- the same per-span cost that
    makes MAP under-segment, reintroduced on top of marginals computed to remove it.
    Measured on a trained checkpoint it decoded to 0.44 of the reference note count
    where MAP-plus-bias reached 0.98. `sum of (P - 1/2)` scores expected *agreement*,
    which has no length preference of its own.

    Pinned as a difference rather than a value: linear means two spans 0.8 apart in
    posterior are 0.8 apart in score, whatever the offset."""
    dmax = 3
    grid = Grid(dmax, torch.zeros(dmax + 1, dtype=torch.long))
    m = torch.zeros(1, 5, dmax, 2)
    m[0, 3, dmax - 1, NOTE] = 0.9          # span [2, 3)
    m[0, 3, dmax - 2, NOTE] = 0.1          # span [1, 3)
    sc = MarginalScorer(m, grid, dmax_s=4.0)

    likely = sc.unary(None, grid, 0, 2, 3)[NOTE]
    unlikely = sc.unary(None, grid, 0, 1, 3)[NOTE]
    assert float(likely - unlikely) == pytest.approx(0.8, abs=1e-6)
    # ...and a span the model thinks more likely absent than present must not pay for
    # itself, or the rule would take every span it is offered.
    assert float(unlikely) < 0 < float(likely)


def test_decoding_from_marginals_returns_a_valid_tiling():
    """The property MEA keeps and thresholded peak-picking would not: the output is
    still a segmentation, so the duration and transition structure the semi-CRF exists
    to impose still holds."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 9, 3
    e = _random_emissions(T, seed=8)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    segs = viterbi(marginal_scorer(dec.scorer, e, grid), e, grid)
    covered = [(i, j) for _, i, j in segs]
    assert covered[0][0] == 0 and covered[-1][1] == T, covered
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:])), covered
    assert all(1 <= j - i <= dmax for i, j in covered), covered


def test_asking_the_log_semiring_for_backpointers_is_refused():
    """The silent version returns an all-zero table, and a traceback over zeros walks
    a real-looking path that no reduction ever chose."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 6, 3
    e = _random_emissions(T, seed=2)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))
    with pytest.raises(ValueError, match="backpointer"):
        scan(
            dec.scorer, e, grid, initial_window(e.onset_logit, 1, dmax), stop=T,
            witness=e.onset_logit.new_zeros((T + 1, 1, 2), dtype=torch.long),
        )


# --------------------------------------------------------------------------- #
# note_bias: correcting max for what sum accounted for                         #
# --------------------------------------------------------------------------- #
def _notes_in(segs):
    return sum(1 for label, _, _ in segs if label == NOTE)


def test_note_bias_equals_brute_force_argmax_over_biased_scores():
    """The same standard the unbiased recurrence is held to. A bias folded into the
    scan accumulates through the window and the backpointers, so "it shifts the answer"
    is not evidence it shifts it by ``k * bias`` -- only enumeration is."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax, bias = 8, 3, 0.8
    e = _random_emissions(T, seed=1)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    scored = [
        (_score_by_loop(dec.scorer, e, s, grid).item() + bias * _notes_in(s), s)
        for s in _all_segmentations(T, dmax)
    ]
    best_score, best = max(scored, key=lambda kv: kv[0])
    got = viterbi(dec.scorer, e, grid, note_bias=bias)
    got_score = _score_by_loop(dec.scorer, e, got, grid).item() + bias * _notes_in(got)
    assert abs(got_score - best_score) < 1e-4, (
        f"biased viterbi scored {got_score}, best possible {best_score}"
    )
    assert got == best, f"biased viterbi returned {got}, brute force says {best}"


def test_a_rest_never_collects_the_bias():
    """It is a *note* insertion bonus. Paid on rests too it would be a constant per
    span, which cancels between any two segmentations of equal span count and so
    corrects nothing -- while still perturbing the decode enough to look like it works
    on a single track."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 6, 3
    e = _random_emissions(T, seed=4)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    # Drive every span to rest with a large negative bias, then confirm a further
    # change of bias cannot move a segmentation that collects none of it.
    all_rest = viterbi(dec.scorer, e, grid, note_bias=-50.0)
    assert _notes_in(all_rest) == 0, "a -50 bias should leave no note spans at all"
    assert viterbi(dec.scorer, e, grid, note_bias=-500.0) == all_rest


def test_note_bias_defaults_to_no_correction():
    """Every result in docs/experiments.md was decoded without one. A default that
    shifted the decode would silently restate all of them."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 8, 3
    e = _random_emissions(T, seed=2)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))
    assert viterbi(dec.scorer, e, grid) == viterbi(dec.scorer, e, grid, note_bias=0.0)


def test_note_bias_moves_note_count_monotonically():
    """``score(y) + b * k(y)`` has increasing differences in ``(b, k)``, so the argmax's
    note count is non-decreasing in ``b``. Worth pinning because it is what makes a
    sweep interpretable: a non-monotone knob has no "the count crosses gold here", and
    the F1 curve's peak would not mean what the script reads it as."""
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 10, 3
    e = _random_emissions(T, seed=3)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    counts = [
        _notes_in(viterbi(dec.scorer, e, grid, note_bias=b))
        for b in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0)
    ]
    assert counts == sorted(counts), f"note count not monotone in the bias: {counts}"
    assert counts[0] < counts[-1], f"the bias moved nothing across +-3: {counts}"


def test_batched_log_partition_equals_brute_force():
    # The batched scan is what training actually runs; check it against ground truth
    # too, not only against the single-track version.
    dec = _fitted_scorer(SemiCRFDecoder())
    T, dmax = 7, 3
    es = [_random_emissions(T, seed=s) for s in (2, 3)]
    ev = _stack_evidence(es)
    grid = Grid(dmax, torch.arange(dmax + 1).clamp(max=dec.scorer.dur.numel() - 1))

    segs = list(_all_segmentations(T, dmax))
    brute = torch.stack([
        torch.logsumexp(
            torch.stack([_score_by_loop(dec.scorer, e, s, grid) for s in segs]), dim=0
        )
        for e in es
    ])
    got = log_partitions(dec.scorer, ev, grid)
    assert torch.allclose(got, brute, atol=1e-4), f"{got.tolist()} != {brute.tolist()}"


def _chunks(e, size, dmax):
    """Split one track's Evidence into fixed-size chunks, each prefixed with up to
    ``dmax`` frames of left context.

    The context is not optional: the alpha carry covers a boundary-crossing span's
    *history*, but the span still reads cue values (``onset[i]``, ``Vc[j]-Vc[i]``) from
    before the chunk. Cumsums are re-based at each chunk's own frame 0, so only
    differences survive and the absolute offset never matters.

    Yields ``(chunk, context)`` -- ``context`` is short only for chunks whose overlap
    already reaches global frame 0."""
    out = []
    for s in range(0, e.T - size + 1, size):
        lo = max(0, s - dmax)
        out.append((
            Evidence(
                e.onset_logit[:, lo : s + size],
                e.voiced_c[:, lo : s + size + 1] - e.voiced_c[:, lo : lo + 1],
                e.rest_c[:, lo : s + size + 1] - e.rest_c[:, lo : lo + 1],
                [s + size - lo],
                [e.frame_times[0][lo : s + size]],
                e.contours,
            ),
            s - lo,
        ))
    return out


def _run_chunked(dec, e, size, grid):
    """Scan a track chunk by chunk, carrying alpha. Returns the cumulative log-mass."""
    carry, A = None, None
    for chunk, ctx in _chunks(e, size, grid.dmax):
        A, carry = chunk_log_partition(dec.scorer, chunk, grid, carry, ctx)
    return A


def _chunks_with_right(e, size, dmax, right):
    """``_chunks``, but each window also carries ``right`` trailing frames.

    That margin exists for the *encoder* -- the TCN's receptive field would otherwise
    see zero-padding at the chunk's right edge -- and the DP must not scan it. Yields
    ``(chunk, context, right)``; the last chunk's margin is clipped to the track."""
    out = []
    for s in range(0, e.T - size + 1, size):
        lo = max(0, s - dmax)
        hi = min(e.T, s + size + right)
        out.append((
            Evidence(
                e.onset_logit[:, lo:hi],
                e.voiced_c[:, lo : hi + 1] - e.voiced_c[:, lo : lo + 1],
                e.rest_c[:, lo : hi + 1] - e.rest_c[:, lo : lo + 1],
                [hi - lo],
                [e.frame_times[0][lo:hi]],
                e.contours,
            ),
            s - lo,
            hi - (s + size),
        ))
    return out


@pytest.mark.parametrize("n_chunks", [2, 5])
def test_chunked_carry_equals_the_whole_track_log_partition(n_chunks):
    # The property the whole truncated-BPTT design rests on: carrying the alpha window
    # across chunk boundaries reproduces the whole-track log Z exactly. If this drifts,
    # chunked training optimises a different objective than it reports.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax

    size = e.T // n_chunks
    trimmed = _truncate(e, size * n_chunks)  # chunks must tile it exactly
    whole = log_partitions(dec.scorer, trimmed, grid)
    got = _run_chunked(dec, trimmed, size, grid)
    assert torch.allclose(got, whole, atol=1e-3), f"{got.item()} != {whole.item()}"


def test_chunked_carry_holds_for_chunks_shorter_than_dmax():
    # dmax spans several chunks here, so a span's candidate starts reach back through
    # more than one boundary -- where an off-by-one in the window or a short context
    # shows up and a single-boundary test would not.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax

    size = max(1, dmax // 3)
    n = e.T // size
    assert n > 3 and size < dmax, "the point is several chunks inside one dmax"
    trimmed = _truncate(e, size * n)
    whole = log_partitions(dec.scorer, trimmed, grid)
    got = _run_chunked(dec, trimmed, size, grid)
    assert torch.allclose(got, whole, atol=1e-3), f"{got.item()} != {whole.item()}"


@pytest.mark.parametrize("right", [1, 11, 40])
def test_a_right_margin_is_not_scanned(right):
    # The encoder needs frames past the chunk's end (its receptive field is ~22 DP
    # frames wide) but the DP must stop at the chunk boundary, or the carry lands
    # `right` frames ahead of where the next chunk resumes and those frames are
    # scanned twice. Every shape still matches when that happens and the loss still
    # falls, so nothing but this equality catches it.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = log_partitions(dec.scorer, trimmed, grid)

    carry, got = None, None
    for chunk, ctx, rgt in _chunks_with_right(trimmed, size, grid.dmax, right):
        got, carry = chunk_log_partition(dec.scorer, chunk, grid, carry, ctx, rgt)
    assert torch.allclose(got, whole, atol=1e-3), f"{got.item()} != {whole.item()}"


def test_scanning_the_right_margin_corrupts_the_carry():
    # The negative control for the above: passing right=0 while the window *has* a
    # right margin double-counts those frames. Pinned so nobody "simplifies" the
    # parameter away on the grounds that the tests pass without it.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = log_partitions(dec.scorer, trimmed, grid)

    carry, got = None, None
    for chunk, ctx, _rgt in _chunks_with_right(trimmed, size, grid.dmax, 11):
        got, carry = chunk_log_partition(dec.scorer, chunk, grid, carry, ctx)  # right=0
    assert not torch.allclose(got, whole, atol=1e-3)


def test_dropping_the_left_context_changes_the_answer():
    # Guards the invariant the docstring states: the carry alone is not enough. Without
    # the overlap the scan clamps boundary-crossing starts to the chunk's own frame 0
    # and scores them against the wrong cues -- silently, so it needs an explicit test.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = log_partitions(dec.scorer, trimmed, grid)

    carry, A = None, None
    for chunk, _ in _chunks(trimmed, size, 0):  # dmax=0 -> no overlap at all
        A, carry = chunk_log_partition(dec.scorer, chunk, grid, carry, 0)
    assert not torch.allclose(A, whole, atol=1e-3), (
        "dropping the left context still matched, so the parity tests above prove "
        "nothing about the context"
    )


def test_detaching_the_carry_still_reaches_every_chunk():
    # Truncated BPTT: with the carry detached, a single end-of-track loss would only
    # reach the last chunk. The telescoping per-chunk loss must put gradient on every
    # chunk's own cues.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    size = e.T // 4
    trimmed = _truncate(e, size * 4)

    grads, carry, prev_A = [], None, 0.0
    for chunk, ctx in _chunks(trimmed, size, grid.dmax):
        onset = chunk.onset_logit.detach().clone().requires_grad_(True)
        live = Evidence(
            onset, chunk.voiced_c, chunk.rest_c,
            chunk.lengths, chunk.frame_times, chunk.contours,
        )
        A, carry = chunk_log_partition(dec.scorer, live, grid, carry, ctx)
        (A - prev_A).sum().backward()
        carry, prev_A = carry.detach(), A.detach()
        grads.append(onset.grad)

    assert len(grads) == 4
    for k, g in enumerate(grads):
        assert g is not None, f"chunk {k} got no gradient"
        assert g.abs().sum() > 0, f"chunk {k} gradient is all zero"


def test_telescoping_chunk_losses_sum_to_the_whole_track_log_partition():
    # The training loss is sum_k (A_k - A_{k-1}); with A_0 = 0 that must telescope to
    # the whole-track log Z, or the reported NLL is not the NLL being optimised.
    dec = _fitted_scorer(SemiCRFDecoder())
    e = _long_emissions(dec)
    grid = dec.scorer.grid(e.frame_times[0])
    dmax = grid.dmax
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = log_partitions(dec.scorer, trimmed, grid)

    total, carry, prev_A = 0.0, None, 0.0
    for chunk, ctx in _chunks(trimmed, size, grid.dmax):
        A, carry = chunk_log_partition(dec.scorer, chunk, grid, carry, ctx)
        total = total + (A - prev_A)
        carry, prev_A = carry.detach(), A.detach()
    assert torch.allclose(total, whole, atol=1e-3), f"{total.item()} != {whole.item()}"


def test_chunk_log_partition_rejects_ragged_lengths():
    # A chunk reads its answer off the end of the window, so a short track would be
    # read at the wrong position rather than failing loudly.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    ea = build_evidence(a, dec.stride)
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    grid = dec.scorer.grid(ea.frame_times[0])
    dmax = grid.dmax
    ragged = build_evidence([a, short], dec.stride)
    with pytest.raises(ValueError, match="uniform lengths"):
        chunk_log_partition(dec.scorer, ragged, grid)


def test_nll_is_differentiable_and_trains_down():
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)

    first = dec.nll(cues, notes)
    assert first.requires_grad
    for _ in range(60):
        opt.zero_grad()
        loss = dec.nll(cues, notes)
        loss.backward()
        # a learned potential receives gradient -> the encoder would too, end to end
        assert dec.scorer.w_on.grad is not None
        opt.step()
    assert dec.nll(cues, notes).item() < first.item()


def test_nll_grad_reaches_the_encoder_through_tensor_cues():
    # A stand-in "encoder": a learnable bias added to the onset logits. If the CRF
    # loss puts gradient on it through TensorCues, the same path reaches a real
    # backbone's heads -- this is what makes end-to-end training possible.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    onset_logit = torch.logit(torch.from_numpy(cues.onset).clamp(1e-6, 1 - 1e-6))
    voicing_logit = torch.logit(torch.from_numpy(cues.voicing).clamp(1e-6, 1 - 1e-6))
    bias = torch.zeros(1, requires_grad=True)  # the "encoder parameter"

    tcues = TensorCues.from_logits(
        cues.contour, onset_logit + bias, cues.frame_times, voicing_logit
    )
    dec.nll(tcues, notes).backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0


def test_numpy_cues_nll_does_not_reach_the_inputs():
    # The inference path: a numpy Cues is a constant, so the loss is differentiable
    # only in the CRF's own potentials -- never the (already-computed) cues.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_evidence(cues, dec.stride)
    assert not e.onset_logit.requires_grad


def test_decode_recovers_two_notes_after_fitting():
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(120):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()

    decoded = dec.decode(cues)
    assert len(decoded) == 2
    onsets = sorted(n.onset for n in decoded)
    assert abs(onsets[0] - 0.2) < 0.1 and abs(onsets[1] - 1.2) < 0.1
    # pitch read from the contour, not invented: ~A2 (110 Hz) then ~D3 (146 Hz)
    pitches = sorted(n.midi_pitch for n in decoded)
    assert abs(pitches[0] - 45) < 1.5 and abs(pitches[1] - 50) < 1.5


def test_batched_nll_equals_single_track_padded():
    # The batched scan over a padded minibatch must equal the per-track scan, track
    # for track -- padding (different lengths) must not leak into a track's log Z.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    na = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    # a shorter track, so the batch pads
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    ns = _notes([(0.2, 0.9)], [45])

    batched = dec.nll_batch([a, short], [na, ns])
    single = torch.stack([dec.nll(a, na), dec.nll(short, ns)])
    assert torch.allclose(batched, single, atol=1e-4)


def test_batched_nll_grad_survives_padding():
    # The padded-batch build scatters graph tensors into a zeros buffer; check grad
    # still reaches an encoder stand-in through the batched scan.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    bias = torch.zeros(1, requires_grad=True)
    onset_logit = torch.logit(torch.from_numpy(a.onset).clamp(1e-6, 1 - 1e-6)) + bias
    voicing_logit = torch.logit(torch.from_numpy(a.voicing).clamp(1e-6, 1 - 1e-6))
    tcues = TensorCues.from_logits(a.contour, onset_logit, a.frame_times, voicing_logit)

    short = _two_note_cues([(0.3, 0.8)], [98.0])
    dec.nll_batch(
        [tcues, short], [_notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), _notes([(0.3, 0.8)], [43])]
    ).mean().backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0


# --- device placement -------------------------------------------------------------
# The semi-CRF had never run off CPU: both entry points that build a LearnedEncoder
# hardcode device="cpu", and there is no semi-CRF training entry point yet, so nothing
# ever called .to() on the potentials. Two fields were stranded on the host as a
# result -- the dur_edges_s buffer (read via .numpy(), which raises once moved) and Jc
# (built from the numpy contour). These tests are what keep that from returning.


def _tensor_cues(cues: Cues, device, bias: torch.Tensor | None = None) -> TensorCues:
    """``cues`` as the grad-bearing TensorCues an encoder on ``device`` would emit.
    ``bias`` stands in for an encoder parameter, so grad has somewhere to land."""
    onset_logit = torch.logit(torch.from_numpy(cues.onset).clamp(1e-6, 1 - 1e-6)).to(device)
    voicing_logit = torch.logit(torch.from_numpy(cues.voicing).clamp(1e-6, 1 - 1e-6)).to(device)
    if bias is not None:
        onset_logit = onset_logit + bias
    return TensorCues.from_logits(
        cues.contour, onset_logit, cues.frame_times, voicing_logit
    )


def _fitted(cues: Cues, notes: list[NoteEvent], steps: int = 120) -> SemiCRFDecoder:
    dec = SemiCRFDecoder()
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(steps):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()
    return dec


def test_numpy_cues_decode_on_device(device):
    # The offline-eval path: numpy Cues have no device of their own, so the module's
    # placement is the authority and the emissions must follow the parameters.
    dec = SemiCRFDecoder().to(device)
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    nll = dec.nll(cues, notes)
    assert nll.device.type == device.type
    assert isinstance(dec.decode(cues), list)


def test_tensor_cues_grad_reaches_encoder_on_device(device):
    # The training path: heads on `device` feeding a decoder on `device`, end to end.
    dec = SemiCRFDecoder().to(device)
    bias = torch.zeros(1, device=device, requires_grad=True)
    tcues = _tensor_cues(_two_note_cues(SPANS, HZ), device, bias)
    dec.nll(tcues, _notes(SPANS, MIDI)).backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0
    assert dec.scorer.w_on.grad is not None


def test_nll_batch_on_device(device):
    # The batched scan is what SemiCRFLoss actually calls, so it needs its own sweep.
    dec = SemiCRFDecoder().to(device)
    full = _two_note_cues(SPANS, HZ)
    short = _two_note_cues([(0.3, 0.8)], [98.0])
    out = dec.nll_batch(
        [_tensor_cues(full, device), _tensor_cues(short, device)],
        [_notes(SPANS, MIDI), _notes([(0.3, 0.8)], [43])],
    )
    assert out.shape == (2,) and out.device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_nll_matches_cpu():
    # Parity, not just liveness: the same potentials must score the same track the
    # same on either backend, or a "device fix" has changed the model.
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    cpu = SemiCRFDecoder()
    gpu = SemiCRFDecoder()
    gpu.load_state_dict(cpu.state_dict())
    assert torch.allclose(cpu.nll(cues, notes), gpu.to("cuda").nll(cues, notes).cpu(),
                          atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_decode_matches_cpu():
    # Viterbi picks discrete boundaries, so a fitted model must decode to the *same*
    # notes on both backends -- onsets are read off the frame grid, not interpolated.
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    cpu = _fitted(cues, notes)
    gpu = SemiCRFDecoder()
    gpu.load_state_dict(cpu.state_dict())
    gpu = gpu.to("cuda")

    on_cpu, on_gpu = cpu.decode(cues), gpu.decode(cues)
    assert len(on_cpu) == len(on_gpu) == 2
    for a, b in zip(on_cpu, on_gpu):
        assert a.onset == pytest.approx(b.onset)
        assert a.offset == pytest.approx(b.offset)
        assert a.midi_pitch == pytest.approx(b.midi_pitch, abs=1e-4)


def test_evidence_rejects_fields_on_different_devices():
    # The invariant's guard. A "meta" tensor is device-distinct from CPU without
    # needing an accelerator, so this runs everywhere -- including the laptop where
    # the original bug was invisible.
    t, cum = torch.zeros(1, 4), torch.zeros(1, 5)
    with pytest.raises(ValueError, match="span multiple devices"):
        Evidence(
            t, cum, torch.zeros(1, 5, device="meta"),
            [4], [np.zeros(4)], [Contour.empty()],
        )


def test_decode_empty_contour_is_no_notes():
    empty = Cues(
        contour=Contour.empty(),
        onset=np.zeros(0, dtype=np.float32),
        frame_times=np.zeros(0),
        voicing=None,
    )
    assert SemiCRFDecoder().decode(empty) == []


def test_decode_without_voicing_head_falls_back_to_contour():
    # No voicing curve: voicing derives from the contour's own voiced mask.
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    cues = Cues(contour=cues.contour, onset=cues.onset, frame_times=cues.frame_times)
    dec = SemiCRFDecoder()
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(120):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()
    assert len(dec.decode(cues)) == 2


def test_the_dp_runs_with_no_contour_at_all():
    # The point of removing the pyin span features: no potential reads a contour, so
    # the whole scan must run without one. If any code path still reaches for it, this
    # raises rather than quietly scoring a feature nobody asked for.
    T = 60
    ft = np.arange(T) * DT
    rng = np.random.default_rng(0)
    cues = TensorCues(
        contour=None,
        onset=torch.sigmoid(torch.from_numpy(rng.standard_normal(T)).float()),
        frame_times=ft,
        voicing=torch.sigmoid(torch.from_numpy(rng.standard_normal(T)).float()),
    )
    dec = SemiCRFDecoder()
    notes = _notes([(0.1, 0.4), (0.5, 0.6)], [45, 50])
    loss = dec.nll(cues, notes)
    assert torch.isfinite(loss)
    loss.backward()   # and the gradient still reaches the cues


def test_decoding_without_a_contour_says_why():
    # The DP scores spans with no contour, but a NoteEvent needs a pitch. That has to
    # fail with the reason, not an IndexError three frames deep.
    T = 30
    cues = TensorCues(
        contour=None,
        onset=torch.zeros(T),
        frame_times=np.arange(T) * DT,
        voicing=torch.full((T,), 0.9),
    )
    with pytest.raises(ValueError, match="no contour"):
        SemiCRFDecoder().decode(cues)


def test_no_voicing_head_and_no_contour_is_refused():
    # Voiced/unvoiced is the one signal the scan cannot proceed without: with neither
    # source, "everything is voiced" would be a plausible number for a question the
    # model was never asked.
    T = 30
    cues = TensorCues(
        contour=None, onset=torch.zeros(T), frame_times=np.arange(T) * DT, voicing=None,
    )
    with pytest.raises(ValueError, match="no voicing signal"):
        SemiCRFDecoder().nll(cues, _notes([(0.1, 0.2)], [45]))


def test_the_learned_pitch_features_survive_into_inference():
    # The DP's `path_length` and `pitch_spread` are functions of the pitch head's
    # per-bin logits. Training passes them through TensorCues; inference builds a numpy
    # Cues. If only one of those carries the logits, a checkpoint decodes as though
    # w_path and w_spread were zero however they were fitted -- and nothing fails,
    # because the discrepancy only exists after training.
    T, K = 40, 216
    ft = np.arange(T) * DT
    rng = np.random.default_rng(0)
    onset = rng.random(T).astype(np.float32)
    voicing = np.full(T, 0.9, dtype=np.float32)
    contour = Contour(times=ft, freqs=np.full(T, 110.0, dtype=np.float32))
    # Confident, and moving halfway through: exactly what the two features detect.
    logits = np.full((T, K), -8.0, dtype=np.float32)
    logits[: T // 2, 40] = 12.0
    logits[T // 2 :, 55] = 12.0

    at_inference = Cues(
        contour=contour, onset=onset, frame_times=ft, voicing=voicing,
        pitch_logits=logits,
    )
    in_training = TensorCues(
        contour=contour,
        onset=torch.from_numpy(onset),
        frame_times=ft,
        voicing=torch.from_numpy(voicing),
        pitch_logits=torch.from_numpy(logits),
    )

    dec = SemiCRFDecoder(stride=1)
    with torch.no_grad():
        dec.scorer.w_path.fill_(3.0)
        dec.scorer.w_spread.fill_(3.0)

    scores = []
    for cues in (at_inference, in_training):
        ev = build_evidence(cues, stride=1, device=torch.device("cpu"))
        assert ev.path_c is not None, "pitch logits did not reach the evidence"
        assert float(ev.path_length(0, 0, T - 1)) > 0.5, "a real pitch move read as zero"
        grid = dec.scorer.grid(ev.frame_times[0])
        scores.append(float(dec.scorer.unary(ev, grid, 0, 0, T - 1)[0].detach()))

    assert scores[0] == pytest.approx(scores[1], abs=1e-5), (
        f"inference scores {scores[0]} but training scored {scores[1]} for the same "
        "cues -- the decoder is not the model that was trained"
    )


# --------------------------------------------------------------------------- #
# Evidence's optional fields, which are same-shaped and were passed positionally #
# --------------------------------------------------------------------------- #
def test_one_preserves_each_optional_field_separately():
    """`path_c` and `entropy_c` are both (B, T+1). Built positionally, transposing them
    type-checks, runs, trains, and scores pitch instability with the entropy cumsum --
    a wrong model that never raises.

    Distinguishable values per field, so a swap fails here rather than months later as
    an unexplained result."""
    B, T, K = 2, 5, 3
    ev = Evidence(
        onset_logit=torch.zeros(B, T),
        voiced_c=torch.full((B, T + 1), 1.0),
        rest_c=torch.full((B, T + 1), 2.0),
        lengths=[T, T],
        frame_times=[np.arange(T) * DT, np.arange(T) * DT],
        contours=None,
        path_c=torch.full((B, T + 1), 3.0),
        entropy_c=torch.full((B, T + 1), 4.0),
        pitch_c=torch.full((B, T + 1, K), 5.0),
    )
    row = ev.one(1)
    assert float(row.voiced_c.mean()) == 1.0
    assert float(row.rest_c.mean()) == 2.0
    assert row.path_c is not None and float(row.path_c.mean()) == 3.0
    assert row.entropy_c is not None and float(row.entropy_c.mean()) == 4.0
    assert row.pitch_c is not None and float(row.pitch_c.mean()) == 5.0


def test_confidence_is_off_unless_asked_for():
    """It costs a second scan with its graph retained -- roughly 3-4x the decode. A
    validation pass wanting only F1 must not pay that, and a note carrying a made-up
    number would be worse than one carrying none: a reviewer sorting by confidence
    would be sorting noise."""
    dec = _fitted_scorer(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    assert all(n.confidence == 1.0 for n in dec.decode(cues))


def test_confidence_is_the_span_posterior():
    """Not a score, not a margin -- the same number `span_marginals` reports, looked up
    for the span that was actually emitted. Pinned against that table directly, because
    an index convention off by one here would still produce plausible spread."""
    dec = _fitted_scorer(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    ev, grid = dec._prepare([cues])
    segs = viterbi(dec.scorer, ev, grid)
    table = span_marginals(dec.scorer, ev, grid)

    got = segment_confidence(table, segs, grid)
    assert got, "no segments to check -- fixture is not exercising the decode"
    for (label, i, j), value in got.items():
        assert value == pytest.approx(
            float(table[0, j, i - j + grid.dmax, label]), abs=1e-6
        )
        assert 0.0 <= value <= 1.0, f"{value} is not a probability"
