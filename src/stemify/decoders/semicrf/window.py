"""The rolling ``alpha`` buffer both recurrences scan over.

``alpha[j][label]`` is the score of the best (Viterbi) or total (forward) way to tile
``[0, j)`` ending in a ``label`` span. The recurrence only ever reads
``alpha[j-dmax:j]``, so nothing older need be kept -- this module is that bounded
buffer and nothing else, shared by ``forward`` and ``viterbi`` so neither depends on
the other.
"""

import torch

from stemify.decoders.semicrf.constants import NEG, START


def new_window(a0: torch.Tensor, dmax: int, dim: int = 0) -> torch.Tensor:
    """A rolling ``alpha`` buffer holding the last ``dmax`` positions, ``a0`` newest.

    Holding them as one tensor is what makes ``prev`` a free *view*: as a Python list,
    every one of the T steps had to ``torch.stack`` up to ``dmax`` separate small
    tensors -- O(T * dmax) copies, ~900k of them on a 5-minute track, which is the
    shape that punishes a GPU hardest (all launch overhead, no arithmetic).

    Positions before 0 are ``NEG``, i.e. unreachable. Two callers rely on that
    differently: ``viterbi`` trims the slice to ``j - lo`` and never reads them, while
    ``forward._scan`` reads the full width every step and lets ``exp(NEG - max)``
    underflow to exactly 0. The second is what keeps its shapes static."""
    shape = list(a0.shape)
    shape.insert(dim, dmax - 1)
    pad = a0.new_full(shape, NEG)
    return torch.cat([pad, a0.unsqueeze(dim)], dim=dim)


def push(window: torch.Tensor, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Slide the window one step, appending ``value`` as the newest position.

    A fresh tensor rather than an in-place write: autograd needs the older alphas
    intact for the backward pass, and mutating a buffer that earlier steps hold views
    into would trip the version counter."""
    head = window.narrow(dim, 1, window.shape[dim] - 1)
    return torch.cat([head, value.unsqueeze(dim)], dim=dim)


def initial_window(ref: torch.Tensor, B: int, dmax: int) -> torch.Tensor:
    """The batched alpha window a track *starts* from: position 0 must be ``START``.

    ``ref`` only supplies dtype and device."""
    a0 = ref.new_full((B, 3), NEG)
    a0[:, START] = 0.0
    return new_window(a0, dmax, dim=1)
