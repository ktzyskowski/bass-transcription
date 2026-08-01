"""What a span is worth: the ``SpanScorer`` contract and its log-linear instance.

The semi-CRF scores a segmentation as ``sum over segments of unary(label, i, j) +
trans(prev -> label)``. This module owns *unary* and *trans*, and is the only place
either is written down.

That exclusivity is the point. The quantity is consumed by three callers with
different access patterns -- the forward scan (every candidate start at a fixed
``j``), Viterbi (the same), and the gold scorer (an arbitrary scattered set of
spans) -- and each used to compute it itself, reaching into the parameters directly.
Four copies of one formula, which have to agree or ``log Z`` and ``score(gold)``
measure different things. When that happens the NLL still falls and the gradients
still flow; the objective simply stops meaning anything, and nothing in the training
loop can see it. One ``unary`` with broadcasting indices serves all three.

``LinearSpanScorer`` is the classical Sarawagi & Cohen form: a handful of scalars
against hand-built features, ``w . f(span)``. A neural scorer is a different
implementation of the same contract, not a different DP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, REST
from stemify.decoders.semicrf.evidence import Evidence


@dataclass(frozen=True)
class Grid:
    """The two grid-derived quantities every DP entry point needs.

    ``dmax`` is the longest representable segment in frames and ``dur_bucket`` maps a
    segment length to its duration-potential bucket. Both fall out of one frame hop
    and the scorer's own configuration, so they are derived together and travel
    together -- they were previously two positional parameters on nine signatures,
    which is the shape that lets a caller pass a ``dur_bucket`` built against a
    different hop than the ``dmax`` beside it.
    """

    dmax: int
    dur_bucket: torch.Tensor


class SpanScorer(nn.Module, ABC):
    """Scores spans and transitions. The DP's only view of the parameters.

    Implementations own *what a segment is worth*; the recurrences own *how the
    segmentations are enumerated*. Nothing in ``forward``/``viterbi``/``score``
    touches a parameter, so a new scorer is a new file rather than an edit spread
    across the DP.
    """

    @abstractmethod
    def unary(
        self,
        ev: Evidence,
        grid: Grid,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        """Score of span ``[i, j)`` in batch row ``b``, for **both** labels.

        ``b``/``i``/``j`` broadcast against each other, and the result is their
        broadcast shape plus a trailing axis of 2 ordered ``(NOTE, REST)``. That is
        what lets one implementation serve both access patterns: the scan passes
        ``b=(B,1)``, ``i=(1,dmax)``, scalar ``j`` and gets ``(B, dmax, 2)``; the gold
        scorer passes three ``(S,)`` tensors and gets ``(S, 2)``.

        Both labels are always computed rather than selected first. It is ~2x the
        arithmetic and a fraction of the cost of branching -- and it keeps the shape
        static, which is what the scan's CUDA-graph capture needs.
        """

    @abstractmethod
    def transitions(self) -> torch.Tensor:
        """``(3, 2)``: rows are the previous label in ``{NOTE, REST, START}``, columns
        the next in ``{NOTE, REST}``. ``START`` is a row rather than a special case so
        the first segment falls out of the same recurrence step as every other one."""

    @abstractmethod
    def grid(self, frame_times: np.ndarray) -> Grid:
        """The ``Grid`` for a frame hop. Derived rather than stored, so one scorer
        decodes cue grids of different resolutions."""


class LinearSpanScorer(SpanScorer):
    """The log-linear scorer: ``w . f(span)`` over hand-built features.

    Five features per ``note`` span -- onset evidence at the start, voiced mass
    across it, interior pitch instability, a bias, and a learned duration bucket --
    and two per ``rest``. Around fifteen scalars in total, which is what makes this
    the *classical* semi-CRF (Sarawagi & Cohen 2004) rather than a neural one: the
    encoder producing the evidence is deep, the thing weighing it is a dot product.
    """

    def __init__(
        self, dmax_s: float = DMAX_S, dur_edges_s: np.ndarray = DUR_EDGES_S
    ) -> None:
        super().__init__()
        self.dmax_s = dmax_s
        self.register_buffer(
            "dur_edges_s", torch.as_tensor(dur_edges_s, dtype=torch.float32)
        )
        self.w_on = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.w_u = nn.Parameter(torch.tensor(1.0))
        self.w_pitch = nn.Parameter(torch.tensor(0.0))  # per interior pitch-change
        self.b_note = nn.Parameter(torch.tensor(0.0))
        self.b_rest = nn.Parameter(torch.tensor(0.0))
        self.dur = nn.Parameter(torch.zeros(len(dur_edges_s) + 1))
        self.trans = nn.Parameter(torch.zeros(2, 2))  # [prev, next] over {note, rest}
        self.init = nn.Parameter(torch.zeros(2))      # START -> {note, rest}

    def grid(self, frame_times: np.ndarray) -> Grid:
        """``dmax`` and the length->bucket map for a frame hop.

        Computed in torch on the module's own device: the map indexes ``self.dur``, so
        it has to land beside it. ``torch.searchsorted`` with ``right=False`` matches
        ``np.searchsorted``'s default ``side='left'`` and both round half-to-even, so
        the bucket edges are bit-for-bit what the numpy version produced."""
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        dmax = max(1, round(self.dmax_s / hop))
        edges = torch.round(self.dur_edges_s / hop).to(torch.int64)
        lengths = torch.arange(dmax + 1, device=edges.device)
        return Grid(dmax, torch.searchsorted(edges, lengths))

    def transitions(self) -> torch.Tensor:
        return torch.stack([self.trans[NOTE], self.trans[REST], self.init])

    def unary(self, ev, grid, b, i, j) -> torch.Tensor:
        note = (
            self.w_on * ev.onset_at(b, i)
            + self.w_v * ev.voiced_mass(b, i, j)
            + self.w_pitch * ev.interior_jumps(b, i, j)
            + self.b_note
            + self.dur[grid.dur_bucket[j - i]]
        )
        rest = self.w_u * ev.rest_mass(b, i, j) + self.b_rest
        return torch.stack([note, rest], dim=-1)
