"""``span_moments`` against the definition it is an O(1) rewrite of.

``moments.py`` had no test of its own. It is the only place in the package where the
implementation and the specification are *different formulas*: the spec is

    M_k(start, end) = sum_{t in [start, end)} phi_t (t - start)^k

which is O(length) per span, and the code computes it from ``order + 1`` prefix arrays
by binomial recombination, which is O(order^2) and independent of length. Everything
that makes that rewrite valid -- the binomial expansion, the sign on ``(-start)``, the
prefix arrays' zero row, the choice of origin -- is invisible to any test comparing the
code against itself, and a wrong one still returns finite numbers of the right shape
that the MLP downstream would simply learn around.

So this file computes the naive sum and compares. That is the whole idea; the rest is
choosing inputs where a plausible mistake actually separates.

Three properties beyond the oracle, each pinning a claim made elsewhere in prose:

**Translation invariance.** ``test_chunked_neural`` says outright that the moments are
"translation-invariant on paper; nothing checked the implementation" -- and chunked
training is exactly where it matters, because the origin becomes the chunk's left edge
rather than the track's start. Here it is checked.

**float64.** The recombination is a difference of quantities of size ``start^k * sum
phi`` producing an answer of size ``(end-start)^(k+1)``, so it cancels catastrophically
at a large origin and a short span. ``ACCUM`` is what makes that survivable, and a test
at a small origin would never notice it being lowered.

**Broadcasting.** ``row``/``start``/``end`` serve the scan's ``(B, dmax)`` grid and the
gold scorer's scattered ``(S,)`` triples through one call, so the two have to agree
span for span.
"""

import math

import pytest
import torch

from stemify.semicrf.moments import ACCUM, moment_cumsums, span_moments

ORDER, WIDTH = 2, 3


def _phi(n_tracks: int, n_frames: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_tracks, n_frames, WIDTH, generator=g)


def _naive(phi: torch.Tensor, row: int, start: int, end: int, order: int) -> torch.Tensor:
    """``span_moments``' definition, summed one frame at a time.

    Deliberately a Python loop over ``float64`` scalars: it shares no code with the
    implementation -- no prefix arrays, no binomial coefficients, no ``torch`` reduction
    -- so agreement is evidence about the rewrite rather than about a shared helper.
    """
    out = []
    length = max(end - start, 1)
    for k in range(order + 1):
        moment = torch.zeros(phi.shape[-1], dtype=ACCUM)
        for t in range(start, end):
            moment += phi[row, t].to(ACCUM) * float((t - start) ** k)
        out.append(moment / float(length ** (k + 1)))
    return torch.cat(out).to(torch.float32)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_every_span_matches_the_naive_sum(order):
    """THE test for this module: every span of every length, against the definition.

    Exhaustive rather than sampled -- there are only ~50 spans at this size, and the
    mistakes this is aimed at (an off-by-one endpoint, a dropped zero row, a sign on
    ``(-start)``) are the kind that show up at one length and not another."""
    n_frames = 10
    phi = _phi(1, n_frames)
    cumsums = moment_cumsums(phi, order)
    for start in range(n_frames):
        for end in range(start + 1, n_frames + 1):
            got = span_moments(cumsums, 0, start, end)
            want = _naive(phi, 0, start, end, order)
            assert torch.allclose(got, want, atol=1e-5), (
                f"order {order}, span [{start}, {end}): {got} != {want}"
            )


def test_the_moments_are_read_at_the_right_endpoints():
    """A half-open span: the frame at ``end`` is *not* in it, the frame at ``start`` is.

    The oracle above would catch a both-ends-shifted read only through the ``(t-start)``
    weighting, which is zero at the first frame for every ``k > 0``. This separates the
    two ends directly by making one frame carry all the signal."""
    n_frames = 8
    phi = torch.zeros(1, n_frames, WIDTH)
    phi[0, 4] = 1.0
    cumsums = moment_cumsums(phi, ORDER)
    # M_0 is plain summation, so it is exactly "is frame 4 inside this span".
    inside = span_moments(cumsums, 0, 2, 5)[:WIDTH] * 3.0     # undo the /length
    excluded = span_moments(cumsums, 0, 2, 4)[:WIDTH]
    starts_here = span_moments(cumsums, 0, 4, 6)[:WIDTH] * 2.0
    assert torch.allclose(inside, torch.ones(WIDTH), atol=1e-6)
    assert torch.allclose(excluded, torch.zeros(WIDTH), atol=1e-6)
    assert torch.allclose(starts_here, torch.ones(WIDTH), atol=1e-6)


def test_a_moment_is_measured_from_the_spans_own_start():
    """``(t - start)^k``, not ``t^k``. The whole point of the recombination.

    Without the ``(-start)`` term the answer would still be a valid prefix difference --
    finite, right shape, growing with position -- and would encode *where in the track*
    the span sits rather than where in the span its energy sits. That is precisely the
    quantity chunking would then break."""
    phi = torch.zeros(1, 12, WIDTH)
    phi[0, 3] = 1.0                       # one frame in, for a span starting at 2
    phi[0, 9] = 1.0                       # one frame in, for a span starting at 8
    cumsums = moment_cumsums(phi, ORDER)
    early = span_moments(cumsums, 0, 2, 6)[WIDTH : 2 * WIDTH]
    late = span_moments(cumsums, 0, 8, 12)[WIDTH : 2 * WIDTH]
    assert torch.allclose(early, late, atol=1e-6), (
        "the same relative position in two places scored differently: M_1 is being "
        "measured from the track's origin, not the span's"
    )
    # ...and it is not simply zero everywhere, which would pass the above vacuously.
    assert float(early.abs().max()) > 0.0


def test_the_moments_survive_a_shift_of_the_origin():
    """Translation invariance, which chunked training depends on and nothing checked.

    Under chunking the prefix sums are built from the *chunk's* frame 0, so a span at
    track position 300 is read at chunk position 40. If the normalised moments were not
    invariant to that, every span crossing a chunk boundary would be scored differently
    from the same span in a whole-track pass -- and the loss would still fall."""
    n_frames, shift = 16, 200
    phi = _phi(1, n_frames, seed=3)
    padded = torch.zeros(1, n_frames + shift, WIDTH)
    padded[0, shift:] = phi[0]

    here = moment_cumsums(phi, ORDER)
    there = moment_cumsums(padded, ORDER)
    for start in range(0, n_frames - 3):
        for end in range(start + 1, min(start + 4, n_frames) + 1):
            near = span_moments(here, 0, start, end)
            far = span_moments(there, 0, start + shift, end + shift)
            assert torch.allclose(near, far, atol=1e-4), (
                f"span [{start}, {end}) scores differently at origin +{shift}: "
                f"max |diff| {float((near - far).abs().max())}"
            )


def test_the_accumulator_is_wide_enough_for_a_far_origin():
    """The reason ``ACCUM`` is float64, stated as a measurement and checked as one.

    The recombination differences two quantities of size ``start^k * sum phi`` to get an
    answer of size ``(end-start)^(k+1)``, losing about ``k * log10(start / (end-start))``
    digits. At order 2, an origin near 250 and a 3-frame span that is ~4.5 digits --
    against float32's 7.2. This runs the same span at both dtypes and asserts the wide
    one is the one that agrees with the naive sum."""
    n_frames, start, end = 300, 250, 253
    phi = _phi(1, n_frames, seed=7)
    want = _naive(phi, 0, start, end, ORDER)

    wide = span_moments(moment_cumsums(phi, ORDER), 0, start, end)
    assert torch.allclose(wide, want, atol=1e-5), (
        f"float64 accumulation still lost the moment: {wide} != {want}"
    )
    # The same arithmetic in float32, to show the margin is real rather than assumed.
    narrow_cumsums = moment_cumsums(phi, ORDER).to(torch.float32).to(ACCUM)
    narrow = span_moments(narrow_cumsums, 0, start, end)
    assert not torch.allclose(narrow, want, atol=1e-5), (
        "float32 prefix sums reproduced the moment exactly, so this fixture no longer "
        "exercises the cancellation ACCUM exists for -- move the origin further out"
    )


def test_the_grid_and_the_scattered_call_agree_span_for_span():
    """One implementation, two access patterns: the scan's ``(B,1) x (1,dmax)`` grid and
    the gold scorer's ``(S,)`` triples. A transposed broadcast is invisible in either
    one alone -- both return the right shape."""
    n_tracks, n_frames, dmax, end = 3, 12, 4, 9
    phi = _phi(n_tracks, n_frames, seed=11)
    cumsums = moment_cumsums(phi, ORDER)

    starts = torch.arange(end - dmax, end).clamp(min=0)
    rows = torch.arange(n_tracks)[:, None]
    grid = span_moments(cumsums, rows, starts[None, :], end)      # (n_tracks, dmax, ·)
    assert grid.shape == (n_tracks, dmax, (ORDER + 1) * WIDTH)

    for row in range(n_tracks):
        for k, start in enumerate(starts.tolist()):
            scattered = span_moments(
                cumsums, torch.tensor([row]), torch.tensor([start]),
                torch.tensor([end]),
            )
            assert torch.allclose(grid[row, k], scattered[0], atol=1e-6), (
                f"row {row}, span [{start}, {end})"
            )
            assert torch.allclose(
                grid[row, k], _naive(phi, row, start, end, ORDER), atol=1e-5
            )


def test_an_empty_span_divides_by_one_rather_than_by_zero():
    """``precompute`` builds a full ``(T+1, dmax)`` table and zeroes the ``end = 0`` row
    rather than skipping it, so a zero-length span is *addressed* even though the scan
    never scores one. It has to return a finite number: a NaN here would propagate
    through the MLP into every span's score, not just this one's."""
    phi = _phi(1, 6, seed=13)
    got = span_moments(moment_cumsums(phi, ORDER), 0, 3, 3)
    assert torch.isfinite(got).all(), got
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)


def test_the_prefix_arrays_carry_a_zero_row():
    """``P[p][row, 0] == 0`` is what makes a span starting at frame 0 read the same
    difference as any other. Without it every span would be short by the whole prefix
    and the error would be *largest* at the track's start."""
    phi = _phi(2, 7, seed=17)
    cumsums = moment_cumsums(phi, ORDER)
    assert cumsums.shape == (ORDER + 1, 2, 8, WIDTH)
    assert bool((cumsums[:, :, 0] == 0).all())
    # P_0 is the plain running sum; P_1 weights each frame by its absolute index.
    assert torch.allclose(cumsums[0, 0, 4], phi[0, :4].to(ACCUM).sum(0), atol=1e-6)
    weighted = sum(phi[0, t].to(ACCUM) * t for t in range(4))
    assert torch.allclose(cumsums[1, 0, 4], weighted, atol=1e-6)


def test_a_negative_order_is_refused():
    with pytest.raises(ValueError, match="order must be"):
        moment_cumsums(_phi(1, 4), -1)


def test_normalisation_bounds_a_moment_by_the_spans_mean():
    """Each ``M_k`` is divided by ``length^(k+1)``, which is what makes a long span's
    moments comparable to a short one's. ``M_k`` grows like ``mean * length^(k+1)/(k+1)``,
    so the normalised value is bounded by the mean whatever the duration -- without it
    the MLP would read length three times over, once per moment, when length is already
    a feature in its own right."""
    for length in (1, 4, 16, 64):
        phi = torch.ones(1, length + 1, WIDTH)
        got = span_moments(moment_cumsums(phi, ORDER), 0, 0, length)
        for k in range(ORDER + 1):
            block = got[k * WIDTH : (k + 1) * WIDTH]
            assert float(block.max()) <= 1.0 + 1e-6, f"order {k}, length {length}"
            # sum_{t<L} t^k / L^(k+1) -> 1/(k+1) as L grows: the mean, weighted.
            if length == 64:
                assert float(block.mean()) == pytest.approx(
                    1.0 / (k + 1), abs=0.02
                ), f"order {k} is not converging on mean/{k + 1}"


def test_the_recombination_matches_the_binomial_identity_term_by_term():
    """The identity in the module docstring, evaluated independently.

    The oracle above checks the *result*; this checks that the result is arrived at by
    the stated expansion, so the docstring stays a specification rather than a story.
    A sign error on ``(-start)`` at odd ``k`` passes neither, but a *coefficient* error
    at ``k=2`` (using ``C(k,p)`` from the wrong row) would only fail here in isolation.
    """
    phi = _phi(1, 20, seed=19)
    order, start, end = 2, 6, 11
    cumsums = moment_cumsums(phi, order)
    diffs = [cumsums[p][0, end] - cumsums[p][0, start] for p in range(order + 1)]
    for k in range(order + 1):
        by_hand = sum(
            math.comb(k, p) * ((-float(start)) ** (k - p)) * diffs[p]
            for p in range(k + 1)
        ) / float((end - start) ** (k + 1))
        got = span_moments(cumsums, 0, start, end)[k * WIDTH : (k + 1) * WIDTH]
        assert torch.allclose(got.to(ACCUM), by_hand, atol=1e-9), f"order {k}"
