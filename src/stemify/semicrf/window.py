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


def initial_window(
    ref: torch.Tensor,
    B: int,
    dmax: int,
    n_rows: int,
    slot: torch.Tensor | None = None,
    label: torch.Tensor | None = None,
) -> torch.Tensor:
    """The alpha window a scan *starts* from: one reachable (position, label), all
    ``NEG`` elsewhere.

    ``ref`` only supplies dtype and device. ``n_rows`` is the transition matrix's row
    count (``LabelSet.n_rows``), and ``START`` is its last row by construction -- so it
    is indexed as ``-1`` here rather than imported, which is what lets one buffer serve
    an alphabet of any width without this module knowing the width.

    Defaulting to ``(dmax - 1, START)`` is a track's beginning: the scan's first step is
    ``j = start + 1`` and reads window index ``k`` as position ``j - dmax + k``, so the
    last slot is position ``start`` -- "no frames consumed yet".

    ``slot`` and ``label`` move that seed **per row**, which is what lets a scan begin
    partway into a track. Both are needed together and for the same reason: a segment
    of a track begins at a real gold boundary, not at a chunk's edge, and the segment
    before it carried a real label, not ``START``. Seeding at the chunk's edge instead
    puts gold's own tiling outside the partition -- ``log Z`` sums only what is
    reachable while ``score(gold)`` is computed straight off the gold list, so the NLL
    goes negative and unbounded, and the model farms the one span nothing normalises.

    Holding the positions as one tensor is what makes the recurrence's ``prev`` a free
    *view*. As a Python list, every one of the T steps had to ``torch.stack`` up to
    ``dmax`` small tensors -- O(T * dmax) copies, ~900k on a 5-minute track, which is
    the shape that punishes a GPU hardest (all launch overhead, no arithmetic).
    """
    window = ref.new_full((B, dmax, n_rows), NEG)
    rows = torch.arange(B, device=window.device)
    at = (
        torch.full((B,), dmax - 1, dtype=torch.long, device=window.device)
        if slot is None else slot.to(window.device)
    )
    row = (
        torch.full((B,), n_rows - 1, dtype=torch.long, device=window.device)
        if label is None else label.to(window.device)
    )
    if int(at.min()) < 0 or int(at.max()) >= dmax:
        raise ValueError(
            f"seed slots {at.tolist()} fall outside the {dmax}-wide alpha window. A "
            "slot is `position - scan_start - 1 + dmax`, and a gold segment ending "
            "inside the chunk starts at most dmax back, so an out-of-range slot means "
            "the gold pieces and the scan disagree about where the chunk begins"
        )
    window[rows, at, row] = 0.0
    return window


def push(window: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Slide the window one step, appending ``value`` as the newest position.

    A fresh tensor rather than an in-place write: autograd needs the older alphas
    intact for the backward pass, and mutating a buffer that earlier steps hold views
    into would trip the version counter."""
    return torch.cat([window.narrow(1, 1, window.shape[1] - 1), value.unsqueeze(1)], 1)
