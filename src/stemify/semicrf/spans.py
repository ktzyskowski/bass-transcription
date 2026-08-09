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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from stemify.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, REST
from stemify.semicrf.evidence import Evidence
from stemify.semicrf.labelset import BINARY, LabelSet


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

    # The longest representable segment, in seconds. Declared here because it is the
    # *DP's* geometry rather than any one parameterisation's: ``grid`` already has to
    # produce a ``dmax`` from it, and the scan's candidate-start span, the dataset's
    # left context and a checkpoint's recorded geometry are all that same number. It
    # was previously readable only off ``LinearSpanScorer``, so ``SemiCRFRecipe`` --
    # which records it -- reached through ``nn.Module.__getattr__`` and got back
    # ``Tensor | Module``. Duration *bucket edges* are deliberately not here: those
    # are the log-linear scorer's parameterisation, and a neural scorer may model
    # duration continuously with no edges at all.
    dmax_s: float

    labels: LabelSet
    """The alphabet this scorer emits potentials over.

    Declared on the contract because it is what every consumer needs to know a shape:
    the recurrence sizes its alpha window from ``n_rows``, ``unary`` returns
    ``n_labels`` columns, ``transitions`` is ``(n_rows, n_labels)``, and the decoder
    reads a segment's pitch back off its label. A scorer that names pitches and one
    that does not are the same contract at different widths.
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
        """Score of span ``[i, j)`` in batch row ``b``, for **every** label.

        ``b``/``i``/``j`` broadcast against each other, and the result is their
        broadcast shape plus a trailing axis of ``labels.n_labels``. That is what
        lets one implementation serve both access patterns: the scan passes
        ``b=(B,1)``, ``i=(1,dmax)``, scalar ``j`` and gets ``(B, dmax, n_labels)``;
        the gold scorer passes three ``(S,)`` tensors and gets ``(S, n_labels)``.

        Every label is computed rather than selected first. It is more arithmetic
        and a fraction of the cost of branching -- and it keeps the shape static,
        which is what the scan's CUDA-graph capture needs.
        """

    @abstractmethod
    def transitions(self) -> torch.Tensor:
        """``(n_rows, n_labels)``: rows are the previous label, plus ``START``;
        columns the next. ``START`` is a row rather than a special case so the first
        segment falls out of the same recurrence step as every other one."""

    @abstractmethod
    def grid(self, frame_times: np.ndarray) -> Grid:
        """The ``Grid`` for a frame hop. Derived rather than stored, so one scorer
        decodes cue grids of different resolutions."""

    @property
    def config(self) -> dict[str, object]:
        """Everything ``__init__`` takes, as plain data, tagged with a ``kind``.

        On the contract rather than on each scorer's recipe, because a checkpoint
        records *the model it was trained as* and the scorer is half of that. It used
        to be the recipe's job, which meant the recipe had to narrow to
        ``LinearSpanScorer`` and raise for anything else -- so a second scorer could
        not be trained into a reloadable checkpoint at all.

        ``kind`` is what ``build_scorer`` dispatches on. Weights ride in the state
        dict as usual; this is only the shape they load into.

        **Not abstract, because being reloadable is a capability and not part of
        what a span is worth.** A scorer written for an experiment or a test has no
        business declaring a checkpoint format, and requiring one taxed every double
        with a method nothing would ever call. Raising instead keeps the gap explicit
        at the point a checkpoint is actually written.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no checkpoint representation: it does not "
            "record what its __init__ takes, so a checkpoint of it could not be "
            "rebuilt. Give it `config`/`from_config` before training a run you "
            "intend to reload."
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SpanScorer":
        """The inverse of ``config``, beside it so the two cannot drift apart.

        ``Any`` rather than ``object`` because the values are plain JSON off disk and
        every read would otherwise need a narrowing helper. A wrong type raises from
        the ``int``/``float`` conversion, naming the field."""
        raise NotImplementedError(
            f"{cls.__name__} records no config, so there is nothing to rebuild from"
        )


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
        # Two labels: this scorer has no pitch potential, so there is nothing for a
        # wider alphabet to score with. `BINARY`'s integers are exactly NOTE/REST/START,
        # so every checkpoint of this class keeps loading unchanged.
        self.labels = BINARY
        self.register_buffer(
            "dur_edges_s", torch.as_tensor(dur_edges_s, dtype=torch.float32)
        )
        self.w_on = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.w_u = nn.Parameter(torch.tensor(1.0))
        # The two learned-pitch features. Zero-init so a run that adds them starts from
        # the model that did not have them, and they are read as evidence only if the
        # optimiser finds them useful -- and so that a checkpoint with no pitch head,
        # whose evidence returns zeros here, is scored by an unchanged model.
        self.w_path = nn.Parameter(torch.tensor(0.0))    # distribution travelled
        self.w_spread = nn.Parameter(torch.tensor(0.0))  # frames disagreeing
        self.b_note = nn.Parameter(torch.tensor(0.0))
        self.b_rest = nn.Parameter(torch.tensor(0.0))
        self.dur = nn.Parameter(torch.zeros(len(dur_edges_s) + 1))
        self.trans = nn.Parameter(torch.zeros(2, 2))  # [prev, next] over {note, rest}
        self.init = nn.Parameter(torch.zeros(2))      # START -> {note, rest}

    @property
    def dur_edges(self) -> torch.Tensor:
        """The duration-bucket edges, typed.

        ``dur_edges_s`` is a registered buffer so it follows ``.to(device)`` and rides
        in the state dict -- both required, since the bucket map is computed on the
        module's own device and a checkpoint must restore the edges it was fitted
        against. The cost is that ``nn.Module.__getattr__`` is typed ``Tensor |
        Module``, so every read would otherwise need its own cast. One here instead.
        """
        return cast(torch.Tensor, self.dur_edges_s)

    def grid(self, frame_times: np.ndarray) -> Grid:
        """``dmax`` and the length->bucket map for a frame hop.

        Computed in torch on the module's own device: the map indexes ``self.dur``, so
        it has to land beside it. ``torch.searchsorted`` with ``right=False`` matches
        ``np.searchsorted``'s default ``side='left'`` and both round half-to-even, so
        the bucket edges are bit-for-bit what the numpy version produced."""
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        dmax = max(1, round(self.dmax_s / hop))
        edges = torch.round(self.dur_edges / hop).to(torch.int64)
        lengths = torch.arange(dmax + 1, device=edges.device)
        return Grid(dmax, torch.searchsorted(edges, lengths))

    @property
    def config(self) -> dict[str, object]:
        return {
            "kind": "linear",
            "dmax_s": self.dmax_s,
            "dur_edges_s": self.dur_edges.cpu().tolist(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LinearSpanScorer":
        edges = config.get("dur_edges_s")
        return cls(
            dmax_s=float(config.get("dmax_s", DMAX_S)),
            dur_edges_s=(
                DUR_EDGES_S if edges is None
                else np.asarray(edges, dtype=np.float32)
            ),
        )

    def transitions(self) -> torch.Tensor:
        return torch.stack([self.trans[NOTE], self.trans[REST], self.init])

    def unary(
        self,
        ev: Evidence,
        grid: Grid,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        # `path` and `spread` answer different questions and are both kept: a path
        # length is large for vibrato that returns where it started, a spread is not,
        # and a spread is what says "this span holds two pitches". Supplying both and
        # letting the weights settle is what a CRF is for.
        note = (
            self.w_on * ev.onset_at(b, i)
            + self.w_v * ev.voiced_mass(b, i, j)
            + self.w_path * ev.path_length(b, i, j)
            + self.w_spread * ev.pitch_spread(b, i, j)
            + self.b_note
            + self.dur[grid.dur_bucket[j - i]]
        )
        rest = self.w_u * ev.rest_mass(b, i, j) + self.b_rest
        return torch.stack([note, rest], dim=-1)
