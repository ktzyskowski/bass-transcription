"""What a span is worth: the ``SpanScorer`` contract.

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

The contract carries no implementation. ``NeuralSpanScorer`` is the one in the tree;
a classical log-linear form (Sarawagi & Cohen, ``w . f(span)`` over hand-built
features) lived here until it was retired, and its going cost the DP nothing -- which
is the property this seam exists to have.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.labelset import LabelSet


class SpanScorer(nn.Module, ABC):
    """Scores spans and transitions. The DP's only view of the parameters.

    Implementations own *what a segment is worth*; the recurrences own *how the
    segmentations are enumerated*. Nothing in ``forward``/``viterbi``/``score``
    touches a parameter, so a new scorer is a new file rather than an edit spread
    across the DP.
    """

    # The longest representable segment, in seconds. Declared here because it is the
    # *DP's* geometry rather than any one parameterisation's: ``dmax_for`` already has
    # to produce a frame count from it, and the scan's candidate-start span, the
    # dataset's left context and a checkpoint's recorded geometry are all that same
    # number. It was previously readable only off a concrete scorer, so
    # ``SemiCRFRecipe`` -- which records it -- reached through ``nn.Module.__getattr__``
    # and got back ``Tensor | Module``.
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
        dmax: int,
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
    def dmax_for(self, frame_times: np.ndarray) -> int:
        """The longest representable segment **in frames**, for this frame hop.

        Derived rather than stored, so one scorer decodes cue grids of different
        resolutions -- or refuses to, if its parameters are indexed by a frame count
        and a different hop would read another span's row."""

    @property
    def config(self) -> dict[str, object]:
        """Everything ``__init__`` takes, as plain data, tagged with a ``kind``.

        On the contract rather than on each scorer's recipe, because a checkpoint
        records *the model it was trained as* and the scorer is half of that. It used
        to be the recipe's job, which meant the recipe had to narrow to one concrete
        scorer and raise for anything else -- so a second scorer could not be trained
        into a reloadable checkpoint at all.

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
