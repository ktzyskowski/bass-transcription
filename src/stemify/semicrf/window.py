"""The rolling ``alpha`` buffer both recurrences scan over.

``alpha[j][label]`` is the score of the best (Viterbi) or total (forward) way to tile
``[0, j)`` ending in a ``label`` span. The recurrence only ever reads
``alpha[j-dmax:j]``, so nothing older need be kept -- this module is that bounded
buffer and nothing else.

Shape is ``(B, dmax, n_rows)``, where ``n_rows`` is the alphabet plus ``START`` -- 3 for
the original ``{note, rest}``, 50 for a 48-pitch alphabet. It used to be the literal 3.
A single track is a batch of one, which is what removed the ``dim=0``/``dim=1``
polymorphism this module used to carry.
"""

import torch

from stemify.semicrf.constants import NEG


def initial_window(ref: torch.Tensor, B: int, dmax: int, n_rows: int) -> torch.Tensor:
    """The alpha window a track *starts* from: position 0 is ``START``, and every
    earlier position is ``NEG`` (unreachable).

    ``ref`` only supplies dtype and device. ``n_rows`` is the transition matrix's row
    count (``LabelSet.n_rows``), and ``START`` is its last row by construction -- so it
    is indexed as ``-1`` here rather than imported, which is what lets one buffer serve
    an alphabet of any width without this module knowing the width.

    Holding the positions as one tensor is what makes the recurrence's ``prev`` a free
    *view*. As a Python list, every one of the T steps had to ``torch.stack`` up to
    ``dmax`` small tensors -- O(T * dmax) copies, ~900k on a 5-minute track, which is
    the shape that punishes a GPU hardest (all launch overhead, no arithmetic).
    """
    a0 = ref.new_full((B, n_rows), NEG)
    a0[:, -1] = 0.0                    # START is the last row, whatever the width
    pad = ref.new_full((B, dmax - 1, n_rows), NEG)
    return torch.cat([pad, a0.unsqueeze(1)], dim=1)


def push(window: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Slide the window one step, appending ``value`` as the newest position.

    A fresh tensor rather than an in-place write: autograd needs the older alphas
    intact for the backward pass, and mutating a buffer that earlier steps hold views
    into would trip the version counter."""
    return torch.cat([window.narrow(1, 1, window.shape[1] - 1), value.unsqueeze(1)], 1)
