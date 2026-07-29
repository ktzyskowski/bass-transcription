"""``SemiCRFPotentials``: the ~15 learned scalars and the scores derived from them.

The parameter-bearing ``nn.Module`` the decoder owns and the DP reads. Isolating it
keeps the recurrences (``dp``) pure functions of ``(potentials, emissions)``: the DP
never touches a raw ``nn.Parameter``, only ``note_potentials`` / ``rest_potentials`` /
``trans_to``, so each recurrence is testable against a hand-set potentials object.
"""

import numpy as np
import torch
from torch import nn

from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, REST
from stemify.decoders.semicrf.emissions import Emissions


class SemiCRFPotentials(nn.Module):
    """The learned potentials over ``{note, rest}`` segmentations of a track."""

    def __init__(
        self, dmax_s: float = DMAX_S, dur_edges_s: np.ndarray = DUR_EDGES_S
    ) -> None:
        super().__init__()
        self.dmax_s = dmax_s
        self.register_buffer(
            "dur_edges_s", torch.as_tensor(dur_edges_s, dtype=torch.float32)
        )

        # w_* weight the soft per-frame evidence; the biases set a note/rest baseline;
        # `dur` is the duration table; `trans` is the note/rest 2x2 block and `init`
        # is START -> {note, rest}.
        self.w_on = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.w_u = nn.Parameter(torch.tensor(1.0))
        self.w_pitch = nn.Parameter(torch.tensor(0.0))  # per interior pitch-change
        self.b_note = nn.Parameter(torch.tensor(0.0))
        self.b_rest = nn.Parameter(torch.tensor(0.0))
        self.dur = nn.Parameter(torch.zeros(len(dur_edges_s) + 1))
        self.trans = nn.Parameter(torch.zeros(2, 2))  # [prev, next] over {note, rest}
        self.init = nn.Parameter(torch.zeros(2))       # START -> {note, rest}

    def grid(self, frame_times: np.ndarray) -> tuple[int, torch.Tensor]:
        """``(dmax_frames, dur_bucket)`` for a frame grid: the max segment length in
        frames and the length->bucket map indexing ``self.dur``. Derived from the
        grid's hop rather than stored, so the decoder is agnostic to the exact hop.

        Computed in torch on the module's own device. The bucket map indexes
        ``self.dur``, so it must land beside it; the previous numpy round trip both
        stranded it on CPU and could not run at all once ``dur_edges_s`` -- a buffer,
        and so moved by ``.to()`` -- was off CPU. ``torch.searchsorted`` with
        ``right=False`` is ``np.searchsorted``'s default ``side='left'``, and both
        round half-to-even, so the bucket edges are bit-for-bit the previous ones."""
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        dmax = max(1, round(self.dmax_s / hop))
        edges = torch.round(self.dur_edges_s / hop).to(torch.int64)
        lengths = torch.arange(dmax + 1, device=edges.device)
        return dmax, torch.searchsorted(edges, lengths)

    def trans_to(self) -> torch.Tensor:
        """(3, 2): rows = prev in {note, rest, START}, cols = next in {note, rest}."""
        return torch.stack([self.trans[NOTE], self.trans[REST], self.init])

    def note_potentials(
        self, e: Emissions, idx: torch.Tensor, j: int, dur_bucket: torch.Tensor
    ) -> torch.Tensor:
        """``note``-potential of every span ``[i, j)`` for ``i in idx``: onset
        evidence at ``i``, voicing mass, interior pitch-change penalty, bias and the
        duration term -- vectorised over the candidate starts."""
        d = j - idx
        return (
            self.w_on * e.onset_logit[idx]
            + self.w_v * (e.Vc[j] - e.Vc[idx])
            + self.w_pitch * (e.Jc[j] - e.Jc[idx + 1])  # interior changes only
            + self.b_note
            + self.dur[dur_bucket[d]]
        )

    def rest_potentials(self, e: Emissions, idx: torch.Tensor, j: int) -> torch.Tensor:
        return self.w_u * (e.Uc[j] - e.Uc[idx]) + self.b_rest
