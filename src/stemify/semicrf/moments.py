"""Position-weighted span pooling in O(1): the moments a learned span feature reads.

A span score has to be computable in O(1) after an O(T) pass, because the scan visits
``T * dmax`` spans. That admits exactly one shape -- a prefix state under an operation
with an inverse, read as a difference -- so the only span quantities available are
running totals. What is summed is free, and that is where the modelling happens.

The plain total ``sum phi_t`` is already a *universal* family over permutation-invariant
span functions (Deep Sets: any function of a set is ``rho(sum phi(x))`` for some pair),
so pooling is not the limitation. **Order is.** A sum does not know what order it summed
in, and a note has an attack, a sustain and a decay, while a span holding two notes has
an onset in its middle.

Weighting each frame by its position inside the span fixes that, and polynomial weights
are the family where the *relative* position a span cares about can be recovered from
the *absolute* position a prefix sum knows::

    M_k(start, end) = sum_{t in [start, end)} phi_t (t - start)^k
                    = sum_{p <= k} C(k, p) (-start)^(k-p) [P_p(end) - P_p(start)]

so ``order + 1`` prefix arrays answer every moment up to ``order`` in O(order^2). ``M_0``
is plain mean-pooling; ``M_1`` says whether the span is front- or back-loaded; ``M_2``
whether its energy is concentrated or spread. Chopping the span into thirds by hand is
the same information with the cut points chosen by eye, and one hyperparameter in place
of three.

**Why the prefix sums are float64.** The binomial identity is a difference of two
quantities of size ``start^k * sum phi``, and the answer is of size
``(end-start)^(k+1)``, so it loses about ``k * log10(start / (end-start))`` digits. At
``k=2``, a chunk-local ``start`` near 250 and
a 3-frame span, that is ~4.5 digits against float32's 7.2 -- measured as ~26% error on
the moment, injected into a feature the model then has to learn around. float64 carries
16, leaving the error at 1e-9. The arrays are ``(order+1, B, T+1, d)`` and built once per
batch, so the cost is a fraction of a millisecond and a couple of MB; the span reads cast
back down. This is the whole reason ``order`` is small rather than a free choice.
"""

from __future__ import annotations

import math

import torch

# The dtype the prefix sums are accumulated and differenced in. See the module docstring:
# the binomial recombination is catastrophically cancelling in float32 at order 2.
ACCUM = torch.float64


def moment_cumsums(phi: torch.Tensor, order: int) -> torch.Tensor:
    """``(B, T, d)`` per-frame features -> ``(order+1, B, T+1, d)`` prefix sums.

    Entry ``[p, row, t]`` is ``sum_{s < t} phi[row, s] * s**p``, with a leading zero
    row so a span reads ``P[p][row, end] - P[p][row, start]``. Time is measured from the *evidence's* own
    frame 0, which under chunked training is the chunk's left context -- the origin only
    has to be consistent between the two reads, and a smaller one is better conditioned.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    n_tracks, n_frames, width = phi.shape
    acc = phi.to(ACCUM)
    times = torch.arange(n_frames, device=phi.device, dtype=ACCUM)
    out = []
    for _ in range(order + 1):
        running = torch.cumsum(acc, dim=1)
        out.append(torch.cat([running.new_zeros(n_tracks, 1, width), running], dim=1))
        acc = acc * times[None, :, None]
    return torch.stack(out)


def span_moments(
    cumsums: torch.Tensor,
    row: torch.Tensor | int,
    start: torch.Tensor | int,
    end: torch.Tensor | int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalised relative-position moments over ``[start, end)`` -> ``(..., (order+1)*d)``.

    ``row``/``start``/``end`` broadcast exactly as ``Evidence``'s span accessors do, so one call
    serves the scan's ``(B, dmax)`` grid of candidates and the gold scorer's scattered
    ``(S,)`` spans.

    Each moment is divided by ``(end - start)**(k+1)``, which is what makes the features
    comparable across span lengths: ``M_k`` grows like ``mean * length**(k+1) / (k+1)``,
    so the normalised value is bounded by the mean whatever the duration. Without it a
    long span's moments would dwarf a short one's and the MLP would be reading length
    three times over -- once per moment -- when length is a feature in its own right.
    """
    order = cumsums.shape[0] - 1
    origin = torch.as_tensor(start, device=cumsums.device, dtype=ACCUM)
    length = torch.as_tensor(end, device=cumsums.device, dtype=ACCUM) - origin
    # (order+1, ..., d): the raw prefix differences, before any recombination.
    diffs = [cumsums[p][row, end] - cumsums[p][row, start] for p in range(order + 1)]
    scale = length.clamp_min(1.0).unsqueeze(-1)
    out = []
    for k in range(order + 1):
        acc = torch.zeros_like(diffs[0])
        for p in range(k + 1):
            acc = acc + math.comb(k, p) * ((-origin) ** (k - p)).unsqueeze(-1) * diffs[p]
        out.append(acc / scale ** (k + 1))
    return torch.cat(out, dim=-1).to(dtype)
