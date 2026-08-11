"""``CueScorer``: the ``SpanScorer`` the DP's tests are written against.

Importable as ``from spanscorer import CueScorer`` -- ``tests/`` is on ``sys.path``
because a ``conftest.py`` sits in it, which is what lets ``tests/training`` and
``tests/semicrf`` share this.

**Why a double rather than the real scorer.** ``NeuralSpanScorer`` is the only
implementation in the tree, and it is the wrong thing to test a *recurrence* with: it
pools a backbone's per-frame activations and scores an alphabet of semitones against a
pitch head, so every fixture would have to carry a trunk and a pitch axis before a
single index convention could be checked. What the DP tests need is a scorer that reads
the evidence -- so a mis-indexed span lands on a different number -- and costs three
synthetic curves to build.

**Why it is here and not in ``src``.** It used to be, as ``LinearSpanScorer``: the
classical Sarawagi & Cohen log-linear form, ``w . f(span)`` over hand-built features.
It was retired once both live checkpoints were neural, and this is the part of it worth
keeping. A production scorer kept alive to serve the tests is a model nobody trains and
everybody has to keep working.

The features are chosen to exercise every index convention the recurrence has:

  - ``onset`` at the span's **first frame** -- an endpoint read, so an off-by-one in
    ``candidate_starts`` moves it,
  - ``voiced``/``rest`` mass across ``[i, j)`` -- interior sums read as cumsum
    differences, so a wrong half-open convention double-counts a frame,
  - a scalar per **length** ``j - i``, indexed directly rather than bucketed, so a span
    scored at the wrong width reads an unrelated number,
  - a full transition table including the ``START`` row.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from stemify.semicrf.constants import DMAX_S, NOTE, REST
from stemify.semicrf.evidence import Evidence
from stemify.semicrf.labelset import BINARY
from stemify.semicrf.spans import SpanScorer

# Length slots in the duration parameter. Indexed by `j - i` with no clamp, so a span
# longer than this raises rather than silently folding onto another length -- which is
# what a bucket map would do. Comfortably past any `dmax` the tests use (115 frames).
DUR_SLOTS = 256


class CueScorer(SpanScorer):
    """``w . f(span)`` over the evidence's own cues, on a ``{note, rest}`` alphabet."""

    def __init__(self, dmax_s: float = DMAX_S) -> None:
        super().__init__()
        self.dmax_s = dmax_s
        self.labels = BINARY
        self.w_on = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.w_u = nn.Parameter(torch.tensor(1.0))
        self.b_note = nn.Parameter(torch.tensor(0.0))
        self.b_rest = nn.Parameter(torch.tensor(0.0))
        self.dur = nn.Parameter(torch.zeros(DUR_SLOTS))
        self.trans = nn.Parameter(torch.zeros(2, 2))  # [prev, next] over {note, rest}
        self.init = nn.Parameter(torch.zeros(2))      # START -> {note, rest}

    def dmax_for(self, frame_times: np.ndarray) -> int:
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        return max(1, round(self.dmax_s / hop))

    def transitions(self) -> torch.Tensor:
        return torch.stack([self.trans[NOTE], self.trans[REST], self.init])

    def unary(
        self,
        ev: Evidence,
        dmax: int,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        note = (
            self.w_on * ev.onset_logit[b, i]
            + self.w_v * (ev.voiced_c[b, j] - ev.voiced_c[b, i])
            + self.b_note
            + self.dur[j - i]
        )
        rest = self.w_u * (ev.rest_c[b, j] - ev.rest_c[b, i]) + self.b_rest
        return torch.stack([note, rest], dim=-1)

    @property
    def config(self) -> dict[str, object]:
        return {"kind": "cue", "dmax_s": self.dmax_s}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CueScorer":
        return cls(dmax_s=float(config.get("dmax_s", DMAX_S)))


def fitted(scorer: CueScorer) -> CueScorer:
    """Move the potentials off their symmetric init, so a bug that happens to cancel
    at ``w=1``/``bias=0`` cannot pass.

    **``dur`` must not be affine in the length**, which is the trap a ramp falls into.
    Every segmentation of ``[0, T)`` has lengths summing to ``T``, so under
    ``dur[d] = a*d + b`` the duration term collapses to ``a*T + b*k`` -- a function of
    the segment *count* alone. Every tiling with the same number of segments then scores
    identically, the brute-force argmax is a tie, and a Viterbi returning some other
    member of that tie looks like a bug. A bounded non-affine curve keeps each length
    worth its own number."""
    with torch.no_grad():
        scorer.w_on.fill_(0.7)
        scorer.w_v.fill_(1.3)
        scorer.w_u.fill_(0.4)
        scorer.b_note.fill_(0.25)
        scorer.b_rest.fill_(-0.4)
        scorer.dur.copy_(torch.cos(torch.arange(DUR_SLOTS, dtype=torch.float32) * 1.7))
        scorer.trans.copy_(torch.tensor([[0.3, -0.2], [-0.5, 0.1]]))
        scorer.init.copy_(torch.tensor([0.6, -0.3]))
    return scorer
