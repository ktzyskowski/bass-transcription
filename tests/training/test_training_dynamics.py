"""The chunked objective's *gradient*, which is not the objective's gradient.

Every committed test of the carry runs under ``torch.no_grad()``. That is not a gap in
coverage, it is a blind spot with a shape: ``test_chunk_carry`` proves the per-chunk
losses sum to the true whole-track NLL, and they do -- exactly, to the last bit. What
nothing checked is whether the *derivative* of that sum is the derivative of the NLL,
and it is not.

``SemiCRFLoss`` detaches the alpha window between chunks, so ``d/dtheta`` of chunk *k*'s
term reaches chunk *k*'s encoder activations and stops. The true derivative also flows
back through ``alpha_{k-1}`` into every earlier chunk of the track. The omitted piece is
weighted by how strongly the incoming alpha depends on the parameters, which is small
while the potentials are flat and large once they sharpen -- so the two gradients agree
at initialisation and come apart as the model learns:

    optimiser steps      loss value        cos(g_truncated, g_exact)
              0          identical                     0.9999
             40          identical                     0.4569
             60          identical                    -0.0243

A negative cosine is not a slightly wrong descent direction. It is an *ascent*
direction, and it is what two GPU runs of the neural scorer did: ordinary through epoch
0, then ``nll`` to 1e6 and ``grad_norm`` to 1e7 in epoch 1. Measured at this scale:
truncated bottoms out at 99 and climbs to 311,807 over 200 steps, while the exact
gradient from the same seed descends monotonically to 13.7 with its norm decaying to
0.05.

**The linear scorer is bitten too, and survives it.** Same comparison, same fixture:
truncated bottoms at 96.5 and drifts back to 120; exact reaches 80.6 and is still
falling. Twenty scalars over hand-crafted features cannot sharpen enough for the
omitted term to dominate, so the bias reads as a plateau rather than a divergence. That
is why this went unnoticed for as long as every scorer was log-linear.

The value being *bit-identical* is what made it invisible. Every number a run logs is
correct; only the direction it steps in is wrong.

**The answer is ``--segment-chunks``** (``tests/training/test_segments.py``): bounded
carry chains, undetached within one, so the gradient is the objective's. Truncation
remains the default because every earlier run used it and a checkpoint should reproduce
what it did, not silently better. What is tested here is therefore the *price* of that
default, not an open defect -- dropping ``d(alpha_k)/d(theta)`` is what truncation is,
and no implementation of it agrees with the exact gradient.

Everything here is pointwise in time, so a chunk's frames get identical activations to
the whole track's at the same absolute positions -- which is what makes ``_true_nll``, a
single uninterrupted scan at the live weights, a legitimate reference for what the loop
claims to be minimising.
"""

from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.core.cache import Cache
from stemify.core.conditions import CONDITIONS
from stemify.core.contour import Contour
from stemify.music.notes import NoteEvent
from stemify.core.serde import CachedItem
from stemify.audio.sources import Sources
from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.fit.dataset import Chunk, ChunkDataset, collate
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.semicrf.fit.tracks import train_items
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.neural import NeuralSpanScorer

N_BINS, HOP_S, STRIDE = 12, 0.0116, 3
CHUNK, DMAX = 10, 30
# Long enough that the omitted gradient path has somewhere to reach. Production tracks
# run a median of 53 chunks; the telescoping fixtures use six, which is short enough
# that a truncated gradient is nearly the whole one.
N_CHUNKS = 24
N_DP = CHUNK * N_CHUNKS
N_CQT = N_DP * STRIDE

DP_HOP = HOP_S * STRIDE
DMAX_S = DMAX * DP_HOP                       # so `grid` derives exactly DMAX frames
LABELS = LabelSet.from_bins(N_BINS)          # 4 pitches + rest
CPU = torch.device("cpu")

# Optimiser steps of warm-up before the gradients are compared. The defect is a
# function of how sharp the potentials are, not of the architecture, so a comparison at
# initialisation measures nothing -- the two gradients agree to four decimal places
# there. Sixty steps is where the cosine crosses zero at this scale; forty already
# reads 0.46, so the test is not balanced on a knife edge.
WARMUP_STEPS = 60


@dataclass
class _Item:
    name: str
    root: Path

    def audio(self) -> str:
        return str(self.root / f"{self.name}.wav")

    def sources(self) -> Sources:
        return Sources(bass=self.audio(), mix=self.audio())

    def reference_notes(self) -> list[NoteEvent]:
        span = N_CQT * HOP_S
        # Two pitches alternating, so the transition table and the label readout both
        # have something to learn. A single repeated pitch would leave the interval
        # embedding at its zero init and the whole label path inert.
        return [
            NoteEvent(onset=t, offset=t + 0.25, midi_pitch=40.0 + 2.0 * (k % 2))
            for k, t in enumerate(np.arange(0.05, span - 0.4, 0.35))
        ]


class _NoOpCache(Cache):
    """The fixture writes its CQT directly; pyin never runs on the empty audio."""

    def ensure_features(self, dataset, transform=None, role="bass"):
        return [self.feature_path(i.sources().role(role)) for i in dataset]

    def contour(self, estimator, audio):
        return Contour(
            times=np.arange(N_CQT) * HOP_S, freqs=np.full(N_CQT, 110.0, dtype=np.float32)
        )


class _Trunk(nn.Module):
    """Pointwise in time, deep in frequency: capacity to sharpen, no edge effects.

    Frame *t*'s trunk vector and every head's logits at *t* come from frame *t*'s bins
    alone, so a windowed scan and an uninterrupted one see identical evidence at
    identical absolute frames. A real TCN's edge frames genuinely differ, which would
    make ``_true_nll`` an approximation rather than a reference.
    """

    HIDDEN = 16

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, self.HIDDEN, kernel_size=(n_bins, 1))   # (B, H, 1, T)
        self.body = nn.Sequential(
            nn.GELU(), nn.Conv1d(self.HIDDEN, self.HIDDEN, 1), nn.GELU()
        )
        self.onset = nn.Conv1d(self.HIDDEN, 1, 1)
        self.voicing = nn.Conv1d(self.HIDDEN, 1, 1)
        self.pitch = nn.Conv1d(self.HIDDEN, n_bins, 1)

    def heads_and_hidden(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        h = self.body(self.stem(x).squeeze(2))                           # (B, H, T)
        out = {
            "onset": self.onset(h).transpose(1, 2),                      # (B, T, 1)
            "voicing": self.voicing(h).transpose(1, 2),                  # (B, T, 1)
            "pitch": self.pitch(h).transpose(1, 2),                      # (B, T, F)
        }
        return out, h.transpose(1, 2)                                    # (B, T, H)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads_and_hidden(x)[0]


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dynamics")
    cache = _NoOpCache(tmp / "cache")
    item = _Item("t0", tmp)
    Path(item.audio()).write_bytes(b"")
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N_BINS, N_CQT)).astype(np.float32)
    CachedItem(item.name, feats, {}, np.arange(N_CQT) * HOP_S).save(
        cache.feature_path(item.audio())
    )
    return [item], cache, feats


def _decoder(seed: int = 1234) -> SemiCRFDecoder:
    torch.manual_seed(seed)
    return SemiCRFDecoder(
        stride=STRIDE,
        scorer=NeuralSpanScorer(
            LABELS, trunk_dim=_Trunk.HIDDEN, frame_dim=16, order=2,
            span_dim=32, dur_dim=8, dmax_s=DMAX_S, dmax_frames=DMAX,
        ),
    )


def _loader(items, cache) -> DataLoader:
    return DataLoader(
        ChunkDataset(
            train_items(items, CONDITIONS["clean"], cache),
            chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE, batch_size=1,
            right_context=0, labels=LABELS,
        ),
        batch_size=None,
    )


def _true_nll(model: nn.Module, decoder: SemiCRFDecoder, items, feats) -> float:
    """``log Z - score(gold)`` over the whole track at the *live* weights.

    The quantity the loop claims to be minimising, computed the one way that needs no
    carry at all: a single chunk spanning the track, scanned uninterrupted. Its own
    ``SemiCRFLoss`` -- sharing the decoder, so it reads the same parameters -- keeps this
    probe from writing into any carry it is measuring.
    """
    times = np.arange(N_CQT) * HOP_S
    chunk = Chunk(
        features=feats[None],
        frame_times=times,
        gold=gold_segments(times[::STRIDE], items[0].reference_notes(), DMAX, LABELS),
        y={}, stream=0, reset=True, prev=LABELS.START,
    )
    probe = SemiCRFLoss(decoder)
    batch = collate([chunk], context=0, right=0)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, metrics = probe.compute(model, batch, CPU)
    model.train(was_training)
    return metrics["total"] * probe.scored_frames(batch, batch.features.shape[-1])


def _track_pass(model, decoder, loader, *, detach_carry: bool) -> torch.Tensor:
    """The whole track's summed chunk losses as ONE tensor, ready to backward once.

    ``Trainer`` cannot be used for the untruncated side and that is not an oversight
    here: it backwards each micro-batch, so with the carry undetached chunk *k+1*'s
    backward walks a graph chunk *k*'s backward already freed, and torch raises
    "Trying to backward through the graph a second time". Whole-track backprop needs
    the whole track's loss accumulated first -- which is what this returns, and what
    ``detach_carry=False`` silently cannot deliver through the production loop.
    """
    loss = SemiCRFLoss(decoder, detach_carry=detach_carry)
    total = torch.zeros(())
    for batch in loader:
        total = total + loss.compute(model, batch, CPU)[0]
    return total


def _flat_grad(model, decoder, loader, *, detach_carry: bool) -> tuple[float, torch.Tensor]:
    """``(loss value, gradient)`` of one whole-track pass, as one flat vector."""
    params = list(chain(model.parameters(), decoder.parameters()))
    for p in params:
        p.grad = None
    total = _track_pass(model, decoder, loader, detach_carry=detach_carry)
    value = float(total.detach())
    total.backward()
    return value, torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params
    ])


@pytest.fixture(scope="module")
def sharpened(track):
    """A model trained ``WARMUP_STEPS`` into the region where the bias bites.

    Warmed with the *exact* gradient, so the point being probed is one honest training
    would actually reach -- warming with the truncated one would compare two gradients
    at a point only the broken path visits, which begs the question.

    Module-scoped because the warm-up is the expensive part and both gradient tests
    want the same point. Neither of them steps the optimiser, so sharing is safe.
    """
    items, cache, feats = track
    torch.manual_seed(0)
    model, decoder = _Trunk(), _decoder()
    loader = _loader(items, cache)
    opt = torch.optim.Adam(
        list(chain(model.parameters(), decoder.parameters())), lr=1e-3
    )
    for _ in range(WARMUP_STEPS):
        opt.zero_grad()
        (_track_pass(model, decoder, loader, detach_carry=False) / N_CHUNKS).backward()
        opt.step()
    return model, decoder, loader


def test_the_chunked_loss_reports_the_right_value(sharpened):
    """Detaching the carry changes nothing about the number, at any sharpness.

    The premise of every existing carry test, restated where it can be contrasted: this
    is what ``test_chunk_carry`` establishes and it is *true*. It is also the entire
    reason the gradient defect below is invisible -- the losses, the curves and the
    early-stopping signal are all exactly right while the steps are not.
    """
    model, decoder, loader = sharpened
    truncated, _ = _flat_grad(model, decoder, loader, detach_carry=True)
    exact, _ = _flat_grad(model, decoder, loader, detach_carry=False)
    assert truncated == pytest.approx(exact, rel=1e-9)


def test_detaching_the_carry_rotates_the_gradient_it_does_not_shrink_it(sharpened):
    """The measured price of ``--segment-chunks 0``, and why the flag exists.

    Compared as a *direction*, not a magnitude: a systematically small gradient is a
    learning-rate question, while a rotated one is not descent at all. At a positive
    cosine the truncated step still decreases the objective, however inefficiently; at
    zero it is orthogonal, and below zero every step provably increases the loss it is
    minimising.

    Asserted rather than xfailed, because this is not a bug awaiting a fix: dropping
    ``d(alpha_k)/d(theta)`` is what truncation *is*, and no implementation of it can
    agree with the exact gradient. What was a defect was having no alternative --
    ``detach_carry=False`` could not run through ``Trainer`` at all. Bounded chains are
    that alternative, and this test is the number that says what taking them costs.
    """
    model, decoder, loader = sharpened
    truncated_value, truncated = _flat_grad(model, decoder, loader, detach_carry=True)
    exact_value, exact = _flat_grad(model, decoder, loader, detach_carry=False)
    cos = float(torch.nn.functional.cosine_similarity(truncated, exact, dim=0))
    assert truncated_value == pytest.approx(exact_value, rel=1e-9), (
        "the two must agree on the loss exactly -- that they do is what made the "
        "gradient defect invisible to every value-based test in the suite"
    )
    assert cos < 0.5, (
        f"the truncated gradient points {cos:.4f} along the objective's own, which is "
        "closer than measured (-0.02 at this warm-up) -- if truncation has become "
        "harmless then bounded chains are buying nothing and should go"
    )


def test_the_objective_is_optimisable_when_the_gradient_is_exact(track):
    """The positive control, and the reason the fix is worth building.

    Without this the diagnosis is incomplete: a divergent run is equally consistent with
    "the gradient is truncated" and "this objective cannot be descended at all". Same
    seed, same learning rate, same optimiser steps as the run that reaches 3e5 -- the
    only change is that the gradient reaches through the carry.
    """
    items, cache, feats = track
    torch.manual_seed(0)
    model, decoder = _Trunk(), _decoder()
    loader = _loader(items, cache)
    opt = torch.optim.Adam(
        list(chain(model.parameters(), decoder.parameters())), lr=1e-3
    )
    start = _true_nll(model, decoder, items, feats)
    trace = []
    for _ in range(80):
        opt.zero_grad()
        (_track_pass(model, decoder, loader, detach_carry=False) / N_CHUNKS).backward()
        opt.step()
        trace.append(_true_nll(model, decoder, items, feats))
    detail = f"start {start:.3f} -> " + " ".join(f"{v:.1f}" for v in trace[::10])
    assert np.isfinite(trace).all(), f"the exact gradient went non-finite: {detail}"
    assert trace[-1] < 0.5 * start, (
        f"the exact whole-track gradient did not descend the objective: {detail}"
    )
