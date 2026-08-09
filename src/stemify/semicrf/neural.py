"""``NeuralSpanScorer``: a learned segment potential over the backbone's own features.

The same contract as ``LinearSpanScorer`` and a different body. ``LinearSpanScorer`` is
``w . f(span)`` over five hand-built features -- onset evidence at the start, voiced mass
across it, how far the pitch distribution travelled, how much its frames disagree, a
duration bucket. Each is a fixed function someone chose, and the model's whole freedom is
one scalar per feature.

Here the per-frame function is learned and the readout is a network::

    phi_t   = MLP(trunk_t || onset_t || voiced_t || unvoiced_t)        T evaluations
    P_k     = prefix sums of phi_t * t^k                               one pass
    f(a, b) = MLP(moments(a,b) || phi at the edges || e_dur[b - a])    T * dmax
    s(a,b,y)= <f(a,b), e_y> + bias_y + alpha(a,b) * logprob_y(a,b)     one matmul

Four things it can express that no setting of the linear scorer's weights can:

**Interaction.** ``w . f`` is linear in its features, so "high pitch spread is only
damning on a *short* span" is not a statement it can make at any weights.

**Order inside the span.** Every existing feature is a sum or an endpoint read, both
blind to permutation. ``moments`` carries where in the span its energy sits, which is
what separates one note's attack-and-decay from two notes butted together.

**Interior onset evidence.** ``onset_at(i)`` reads the span's first frame and nothing
else, so "there is a strong onset in the *middle* of this span" -- the single best
signal for rejecting a merged double-note -- had no feature at all. It arrives here as
``M_0`` of the onset channel, and its position as ``M_1``.

**A span-dependent weight on the pitch head.** ``alpha(a, b)`` lets the model distrust
the head across a transient and trust it across a sustain; ``w_path``/``w_spread`` are
single global scalars that weight pitch evidence identically everywhere.

The label enters bilinearly and not as an MLP input. That is not a style choice: at 49
labels and ~600k spans a track, ``g(a, b, y)`` evaluated per label is 29M MLP calls,
while one ``f(a, b)`` and a dot product against an embedding table is 600k plus a matmul.

**What it does not do is pick its own pitch.** ``alpha * logprob_y`` reads the pitch
head's own per-semitone log-probability rather than reconstructing pitch identity inside
``f``'s bottleneck. The head is already calibrated -- it beat pyin outright -- and asking
a 32-dimensional pooled summary to also encode which of 48 semitones is sounding would
spend the capacity twice.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from stemify.semicrf.constants import DMAX_S
from stemify.semicrf.evidence import Evidence
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.moments import moment_cumsums, span_moments
from stemify.semicrf.spans import Grid, SpanScorer

# How many span-end positions to score in one batched MLP call. Bounds the peak
# activation footprint without changing the result: the table is built in slices along
# `j` and concatenated. A training chunk is ~128 positions and fits in one slice, so this
# only binds at inference, where a whole track is thousands of frames and the unsliced
# intermediates would be tens of GB.
J_BLOCK = 256


class NeuralSpanScorer(SpanScorer):
    """A learned span potential: pooled trunk features through an MLP, labels bilinear."""

    def __init__(
        self,
        labels: LabelSet,
        trunk_dim: int,
        frame_dim: int = 32,
        order: int = 2,
        span_dim: int = 64,
        dur_dim: int = 16,
        dmax_s: float = DMAX_S,
        dmax_frames: int = 115,
    ) -> None:
        """``trunk_dim`` is the backbone's hidden width, which the evidence must match.

        ``dmax_frames`` sizes the duration embedding, and it is in *frames* because the
        embedding is indexed by ``b - a``. That pins this scorer to one grid resolution,
        unlike ``LinearSpanScorer``, whose duration edges are in seconds and so survive a
        change of hop. ``grid`` checks rather than adapts: a hop that derives a different
        ``dmax`` would silently index a table fitted against another one.
        """
        super().__init__()
        self.labels = labels
        self.dmax_s = dmax_s
        self.dmax_frames = dmax_frames
        self.order = order
        self.frame_dim = frame_dim
        self.trunk_dim = trunk_dim

        # +3: the onset logit and the two voicing log-probabilities, alongside the trunk.
        # They are functions of the trunk and so add no information, but they are the
        # *supervised* functions of it -- the aux terms train them directly -- so handing
        # them over saves the projection from rediscovering quantities that already exist
        # in calibrated form.
        self.phi = nn.Sequential(
            nn.Linear(trunk_dim + 3, frame_dim),
            nn.GELU(),
            nn.Linear(frame_dim, frame_dim),
        )
        self.dur = nn.Embedding(dmax_frames + 1, dur_dim)
        # (order+1) moments and three edge reads: the frame before the span, its first,
        # and its last. The first two are what a boundary separates, and a boundary is
        # the thing this decoder exists to place.
        span_in = (order + 1) * frame_dim + 3 * frame_dim + dur_dim
        self.f = nn.Sequential(
            nn.Linear(span_in, span_dim),
            nn.GELU(),
            nn.Linear(span_dim, span_dim),
        )
        self.label_emb = nn.Embedding(labels.n_labels, span_dim)
        self.label_bias = nn.Parameter(torch.zeros(labels.n_labels))
        self.alpha = nn.Linear(span_dim, 1)
        # What a `rest` span's pitch term is worth, since it names no semitone to read
        # one for. Started at `-log(n_pitches)`, which is the log-probability an
        # *uninformative* pitch head assigns to every semitone -- so rest begins exactly
        # as good as a pitch nobody can identify, and the head's confidence is what has
        # to earn a note. Zero would have started it better than any note can ever score.
        self.rest_logprob = nn.Parameter(
            torch.full((1,), -math.log(labels.n_pitches))
        )

        # Zero-init the label readout, so an untrained scorer scores every span by its
        # pitch agreement alone -- a sane starting model rather than noise through a
        # 49-way softmax.
        nn.init.zeros_(self.label_emb.weight)
        nn.init.zeros_(self.label_bias)
        nn.init.zeros_(self.dur.weight)
        # `alpha` is deliberately *not* zeroed, and that is load-bearing rather than an
        # omission. With both it and `label_emb` at zero, `d score / d f` is identically
        # zero, so `f`, `phi` and `dur` receive no gradient at all on the first step --
        # the whole network downstream of the trunk is inert until the embedding drifts
        # off zero. Leaving the gate at its default init keeps one live path into `f`
        # from the start, and it is the right one: the only thing `f` can say at init is
        # how far to trust the pitch head on this span.

        # Transitions, factorised. A full (n_rows, n_labels) table is 2450 free entries
        # at this alphabet, nearly all of them (pitch -> pitch) pairs seen a handful of
        # times. What actually generalises is the *interval*: a fifth is a fifth wherever
        # it starts, so one entry per representable interval is transposition-equivariant
        # and needs no clipping constant -- there are exactly 2*n_pitches - 1 of them.
        self.class_trans = nn.Parameter(torch.zeros(3, 2))     # {pitch,rest,START}x{p,r}
        self.interval = nn.Parameter(torch.zeros(2 * labels.n_pitches - 1))

    def grid(self, frame_times: np.ndarray) -> Grid:
        """``dmax`` for a frame hop, checked against the duration table's width."""
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        dmax = max(1, round(self.dmax_s / hop))
        if dmax != self.dmax_frames:
            raise ValueError(
                f"this scorer's duration embedding is sized for dmax={self.dmax_frames} "
                f"frames, but a {hop * 1000:.2f} ms hop over dmax_s={self.dmax_s}s gives "
                f"{dmax}. The embedding is indexed by (b - a) in frames, so decoding on a "
                "different grid than it was fitted on would read another span's row."
            )
        # An empty bucket map: this scorer has no duration *buckets*, it embeds `b - a`
        # directly. `Grid` still carries the field for `LinearSpanScorer`'s sake.
        return Grid(dmax, torch.zeros(dmax + 1, dtype=torch.long))

    def transitions(self) -> torch.Tensor:
        n, rest = self.labels.n_pitches, self.labels.REST
        out = self.class_trans.new_zeros(self.labels.n_rows, self.labels.n_labels)
        pitches = torch.arange(n, device=self.class_trans.device)
        # `interval[q - p + n - 1]`: index 0 is the widest downward leap the alphabet can
        # represent and `2n - 2` the widest upward, so unison sits dead centre.
        idx = pitches[None, :] - pitches[:, None] + (n - 1)
        out[:n, :n] = self.class_trans[0, 0] + self.interval[idx]
        out[:n, rest] = self.class_trans[0, 1]
        out[rest, :n] = self.class_trans[1, 0]
        out[rest, rest] = self.class_trans[1, 1]
        out[self.labels.START, :n] = self.class_trans[2, 0]
        out[self.labels.START, rest] = self.class_trans[2, 1]
        return out

    def _frame_features(self, ev: Evidence) -> torch.Tensor:
        """``(B, T, frame_dim)`` -- the learned per-frame projection ``phi``."""
        if ev.hidden is None:
            raise ValueError(
                "NeuralSpanScorer pools the backbone's per-frame activations, and this "
                "Evidence carries none. Encode through a model with a backbone "
                "(CueNet.heads_and_hidden fills Frames.hidden), or score with "
                "LinearSpanScorer, whose features are all head outputs."
            )
        if ev.hidden.shape[-1] != self.trunk_dim:
            raise ValueError(
                f"trunk width {ev.hidden.shape[-1]} does not match the "
                f"{self.trunk_dim} this scorer was built for; the checkpoint and "
                "the backbone disagree."
            )
        # Per-frame voicing, recovered by differencing its own cumsum rather than carried
        # twice. `voiced_c[t+1] - voiced_c[t]` is exactly `log P(voiced at t)`, which is
        # what went into the sum.
        voiced = ev.voiced_c[:, 1:] - ev.voiced_c[:, :-1]
        rest = ev.rest_c[:, 1:] - ev.rest_c[:, :-1]
        stacked = torch.stack([ev.onset_logit, voiced, rest], dim=-1)
        out: torch.Tensor = self.phi(torch.cat([ev.hidden, stacked], dim=-1))
        return out

    def _span_vectors(
        self,
        phi: torch.Tensor,
        cumsums: torch.Tensor,
        b: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
    ) -> torch.Tensor:
        """``(..., span_dim)`` -- ``f(a, b)``, everything except the label."""
        # Broadcast the three indices *first*, so every part below lands on one
        # shape and the concatenation needs no expanding. Reading `phi[b, j - 1]`
        # against an unbroadcast `j` would give a narrower tensor than the moments
        # beside it -- and `cat` would then reject shapes that are, in the sense
        # that matters, compatible.
        lead = b + i + j                     # a tensor of the common shape; value unused
        b, i, j = b.expand_as(lead), i.expand_as(lead), j.expand_as(lead)
        moments = span_moments(cumsums, b, i, j, dtype=phi.dtype)
        last = phi.shape[1] - 1
        before = phi[b, (i - 1).clamp(min=0)]
        first = phi[b, i.clamp(max=last)]
        final = phi[b, (j - 1).clamp(min=0, max=last)]
        length = (j - i).clamp(0, self.dmax_frames)
        out: torch.Tensor = self.f(
            torch.cat([moments, before, first, final, self.dur(length)], dim=-1)
        )
        return out

    def _label_scores(
        self, f: torch.Tensor, ev: Evidence, b: torch.Tensor,
        i: torch.Tensor, j: torch.Tensor,
    ) -> torch.Tensor:
        """``(..., n_labels)`` -- the bilinear readout plus the gated pitch term."""
        scores = f @ self.label_emb.weight.T + self.label_bias
        logprob = ev.mean_pitch_logprob(b, i, j)                  # (..., n_pitches)
        rest = self.rest_logprob.expand(*logprob.shape[:-1], 1)
        gate = torch.nn.functional.softplus(self.alpha(f))        # (..., 1)
        return scores + gate * torch.cat([logprob, rest], dim=-1)

    def precompute(self, ev: Evidence, grid: Grid) -> torch.Tensor:
        """``(B, T+1, dmax, n_labels)`` -- every span's unary, in one pass.

        Built here rather than per scan step because the DP is launch-bound: calling an
        MLP once per ``j`` is ``T`` launches of a few hundred rows each, which is the
        shape that lost 22x to a graph-captured scan before. One batched call instead,
        and ``unary`` becomes a gather.

        Sliced along ``j`` so the peak activation footprint is bounded by ``J_BLOCK``
        rather than by track length. Result and gradient are identical either way -- the
        spans are independent, and the slices concatenate.
        """
        phi = self._frame_features(ev)
        cumsums = moment_cumsums(phi, self.order)
        dev = phi.device
        offsets = torch.arange(grid.dmax, device=dev)             # b - a - 1
        rows = torch.arange(ev.B, device=dev)[:, None, None]
        # `j = 0` is zeroed rather than computed. No span ends at frame 0 -- the scan
        # runs `j` from 1 and gold spans are half-open -- and the empty span it would
        # describe divides by its own zero length in `mean_pitch_logprob`. Keeping the
        # row makes `j` index the table directly, with no offset to get wrong.
        blocks = [phi.new_zeros(ev.B, 1, grid.dmax, self.labels.n_labels)]
        for lo in range(1, ev.T + 1, J_BLOCK):
            ends = torch.arange(lo, min(lo + J_BLOCK, ev.T + 1), device=dev)
            j = ends[None, :, None]                               # (1, J, 1)
            # Clamped at 0 exactly as `candidate_starts` does, so a span reaching before
            # the evidence pairs with a NEG alpha rather than indexing out of bounds.
            i = (j - offsets[None, None, :] - 1).clamp(min=0)      # (1, J, dmax)
            f = self._span_vectors(phi, cumsums, rows, i, j)
            blocks.append(self._label_scores(f, ev, rows, i, j))
        return torch.cat(blocks, dim=1)

    def unary(
        self,
        ev: Evidence,
        grid: Grid,
        b: torch.Tensor | int,
        i: torch.Tensor | int,
        j: torch.Tensor | int,
    ) -> torch.Tensor:
        """A gather into the precomputed table, which is built on first call.

        Lazily rather than by an explicit setup call, so every DP entry point -- the
        batched scan, the chunked one, Viterbi, the gold scorer -- gets it without each
        having to remember. The table is stored on the ``Evidence``, whose lifetime is
        exactly one batch, so there is no invalidation rule to get wrong.
        """
        if ev.span_scores is None:
            ev.span_scores = self.precompute(ev, grid)
        # `j - i - 1` is the span's length minus one, which is the axis the table is
        # built along. `candidate_starts` clamps `i` at 0, so for `j <= dmax` several
        # candidates collapse onto start 0 and land at `j - 1`, still inside the axis.
        return ev.span_scores[b, j, j - torch.as_tensor(i, device=ev.device) - 1]
