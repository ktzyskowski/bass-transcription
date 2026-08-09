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

    M_k(a, b) = sum_{t in [a, b)} phi_t (t - a)^k
              = sum_{p <= k} C(k, p) (-a)^(k - p) [P_p(b) - P_p(a)]

so ``order + 1`` prefix arrays answer every moment up to ``order`` in O(order^2). ``M_0``
is plain mean-pooling; ``M_1`` says whether the span is front- or back-loaded; ``M_2``
whether its energy is concentrated or spread. Chopping the span into thirds by hand is
the same information with the cut points chosen by eye, and one hyperparameter in place
of three.

**Why the prefix sums are float64.** The binomial identity is a difference of two
quantities of size ``a^k * sum phi``, and the answer is of size ``(b-a)^(k+1)``, so it
loses about ``k * log10(a / (b-a))`` digits. At ``k=2``, a chunk-local ``a`` near 250 and
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

    Entry ``[p, b, t]`` is ``sum_{s < t} phi[b, s] * s**p``, with a leading zero row so
    a span reads ``P[p][b, j] - P[p][b, i]``. Time is measured from the *evidence's* own
    frame 0, which under chunked training is the chunk's left context -- the origin only
    has to be consistent between the two reads, and a smaller one is better conditioned.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    B, T, d = phi.shape
    acc = phi.to(ACCUM)
    t = torch.arange(T, device=phi.device, dtype=ACCUM)
    out = []
    for _ in range(order + 1):
        running = torch.cumsum(acc, dim=1)
        out.append(torch.cat([running.new_zeros(B, 1, d), running], dim=1))
        acc = acc * t[None, :, None]
    return torch.stack(out)


def span_moments(
    cumsums: torch.Tensor,
    b: torch.Tensor | int,
    i: torch.Tensor | int,
    j: torch.Tensor | int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalised relative-position moments over ``[i, j)`` -> ``(..., (order+1)*d)``.

    ``b``/``i``/``j`` broadcast exactly as ``Evidence``'s span accessors do, so one call
    serves the scan's ``(B, dmax)`` grid of candidates and the gold scorer's scattered
    ``(S,)`` spans.

    Each moment is divided by ``(j - i)**(k+1)``, which is what makes the features
    comparable across span lengths: ``M_k`` grows like ``mean * length**(k+1) / (k+1)``,
    so the normalised value is bounded by the mean whatever the duration. Without it a
    long span's moments would dwarf a short one's and the MLP would be reading length
    three times over -- once per moment -- when length is a feature in its own right.
    """
    order = cumsums.shape[0] - 1
    start = torch.as_tensor(i, device=cumsums.device, dtype=ACCUM)
    length = torch.as_tensor(j, device=cumsums.device, dtype=ACCUM) - start
    # (order+1, ..., d): the raw prefix differences, before any recombination.
    diffs = [cumsums[p][b, j] - cumsums[p][b, i] for p in range(order + 1)]
    scale = length.clamp_min(1.0).unsqueeze(-1)
    out = []
    for k in range(order + 1):
        acc = torch.zeros_like(diffs[0])
        for p in range(k + 1):
            acc = acc + math.comb(k, p) * ((-start) ** (k - p)).unsqueeze(-1) * diffs[p]
        out.append(acc / scale ** (k + 1))
    return torch.cat(out, dim=-1).to(dtype)
