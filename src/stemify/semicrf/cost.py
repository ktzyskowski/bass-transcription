"""``Δ(y, gold)``: how wrong a candidate segmentation is, priced per span.

The structured-margin objective is the NLL with a cost added inside the partition::

    L = log sum_y exp( score(y) + Δ(y, gold) )  -  score(gold)
      = NLL  +  log E_P[ exp Δ ]

which is the likelihood plus a penalty on the model's *own expected cost*. That second
term is ``>= 0`` and vanishes exactly when the posterior puts all its mass on zero-cost
structures, so the pressure self-extinguishes where the problem is already solved and
persists on the residue. At small ``margin`` it linearises to ``margin * E_P[cost]`` --
"the NLL plus a few nats per expected boundary error".

**Δ must decompose over segments**, or it cannot ride inside the DP. Everything here is
a per-span number added to that span's unary, which is exactly what ``scan``'s
``perturb`` argument already accepts -- so this needs no change to the recurrence.

**Cost counts boundaries, not labels.** The textbook choice is per-frame Hamming, and it
is wrong for this failure: a span that merges two *same-pitch* notes carries the correct
label on every frame and pays nothing, while it is precisely the error onset evidence
exists to catch. Boundaries price it. The complementary argument is that the auxiliary
terms already supervise labels densely -- pitch gets a cross-entropy on every voiced
frame -- while *nothing except the structured loss can see a boundary at all*, so Δ
should price what only it can.

**The cost is tolerant, because the metric is.** A note counts as found when its onset
lands within 50 ms, so a boundary one frame off the gold one is *correct*, and charging
it would price an error the thing we are measuring does not consider an error. Worse, it
is exactly the region the plateaued onset target leaves the model indifferent about --
the cost would be fighting jitter rather than mistakes. ``tolerance_frames`` is therefore
derived from the note metric's own onset tolerance and the DP hop, not chosen.

Δ(gold, gold) = 0 by construction: every gold segment starts on a gold boundary and
contains none, so ``margin`` at 0 reproduces the plain NLL exactly and any run that
gains a margin starts from the model that did not have one.
"""

from collections.abc import Sequence

import torch

Segmentation = Sequence[tuple[int, int, int]]


def boundary_cost(
    golds: Sequence[Segmentation],
    n_frames: int,
    dmax: int,
    device: torch.device,
    tolerance_frames: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(B, n_frames+1, dmax, 1)`` -- boundary errors of every candidate span.

    Indexed the way the scan enumerates: entry ``[b, j, d]`` is the span ending at ``j``
    whose start is the ``d``-th candidate, matching ``candidate_starts``. The trailing
    axis is 1 rather than ``n_labels`` because the cost is label-blind, and broadcasting
    is what keeps it that way in memory as well as in principle.

    Two errors per span, each a count:

    ``spurious`` -- the span begins where gold has no boundary, so it splits a gold
    segment. ``missed`` -- a gold boundary falls strictly inside it, so it merges two.
    Under-segmentation is the second; over-segmentation the first. Pricing both means a
    margin cannot buy recall by trading it for fragmentation.

    **The window's left edge is charged nothing, and that is deliberate.** Under chunked
    training the caller's gold holds the segments *ending* in the scanned region, so
    their starts reach back into the left context but boundaries deeper than the earliest
    of them are simply not in view -- they belong to the previous chunk's gold. Charging
    a span for starting there would price it against evidence we were not given. Spans
    that begin before the first known boundary are therefore free, which under-penalises
    rather than inventing a penalty; it reaches only spans longer than the distance from
    the chunk's edge, which no bass note is.
    """
    B = len(golds)
    is_boundary = torch.zeros(B, n_frames + 1, device=device, dtype=dtype)
    # Where gold's knowledge starts in each row. `n_frames` (past the end) for an empty
    # gold, which makes every span free rather than every span guilty.
    known_from = torch.full((B,), float(n_frames), device=device, dtype=dtype)
    for b, gold in enumerate(golds):
        gold_starts = [i for _, i, _ in gold]
        if not gold_starts:
            continue
        is_boundary[b, torch.tensor(gold_starts, device=device)] = 1.0
        known_from[b] = float(min(gold_starts))

    # Dilated by the metric's tolerance: "is there a gold boundary *near* here". A
    # max-pool rather than a sum, so two gold boundaries a frame apart still forgive one
    # candidate rather than crediting it twice.
    width = 2 * tolerance_frames + 1
    near = torch.nn.functional.max_pool1d(
        is_boundary[:, None], kernel_size=width, stride=1, padding=tolerance_frames
    )[:, 0]
    # cum[b, t] = number of gold boundaries strictly below t. Raw, not dilated: a gold
    # boundary swallowed by a span is swallowed however close it sits to another.
    cum = torch.cat(
        [is_boundary.new_zeros(B, 1), torch.cumsum(is_boundary, dim=1)], dim=1
    )
    ends = torch.arange(n_frames + 1, device=device)
    offsets = torch.arange(dmax, device=device)
    # The same arithmetic as `candidate_starts`, for every end at once.
    starts = (ends[:, None] - dmax + offsets[None, :]).clamp(min=0)   # (n_frames+1, dmax)

    spurious = 1.0 - near[:, starts]                                 # (B, F+1, dmax)
    # Counted over `(i + tol, j - tol)`: a gold boundary within tolerance of the span's
    # own endpoints is matched *by* those endpoints, not swallowed by the span.
    hi = (ends - tolerance_frames).clamp(0, n_frames + 1)
    lo = (starts + 1 + tolerance_frames).clamp(0, n_frames + 1)
    missed = (cum[:, hi][:, :, None] - cum[:, lo]).clamp(min=0.0)
    known = (starts[None, :, :] >= known_from[:, None, None]).to(dtype)
    cost: torch.Tensor = ((spurious + missed) * known).unsqueeze(-1)
    return cost
