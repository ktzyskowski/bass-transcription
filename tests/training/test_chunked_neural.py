"""The two seams the neural scorer's divergence could be hiding in, neither covered.

``test_chunk_carry`` verifies the strongest invariant in the suite -- summing the
per-chunk telescoping losses reproduces one uninterrupted scan of the whole track -- but
only for ``LinearSpanScorer``. ``NeuralSpanScorer`` has never been run under chunking at
all: every test of it is a single batch, whole-sequence, no carry. That matters because
its potentials are built from *prefix sums over absolute frame index*, and under chunking
the origin is the window's start rather than the track's. The moments are translation-
invariant on paper; nothing checked the implementation.

And ``test_a_second_epoch_gives_the_same_losses`` runs two train passes back to back over
one dataset. Production runs train -> **validation** -> train through a single
``SemiCRFLoss``, and with ``num_workers=0`` both loaders shard to ``base = 0``, so every
val chunk writes its carry onto a train stream's key.

Two runs of the neural scorer diverged in epoch 1 and never in epoch 0, which is what
these two tests are aimed at.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stemify.core.conditions import CONDITIONS
from stemify.core.serde import CachedItem
from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.fit.dataset import Chunk, ChunkDataset, collate
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.neural import NeuralSpanScorer

# `track` is imported to be *injected*, so every test taking it as a parameter reads
# to ruff as a redefinition -- the standard false positive for a fixture shared
# across modules. The real fix is a `conftest.py`; that means moving fixtures out of
# a passing file, which is worth doing when something else brings us here.
from test_chunk_carry import (  # noqa: F401  (fixtures)
    CHUNK, DMAX, HOP_S, N_BINS, N_CQT, STRIDE, _Item, _NoOpCache, track,
)
from stemify.semicrf.fit.tracks import train_items

DP_HOP = HOP_S * STRIDE
DMAX_S = DMAX * DP_HOP                       # so `grid` derives exactly DMAX frames
LABELS = LabelSet.from_bins(N_BINS)          # 4 pitches + rest


class _PointwiseTrunk(nn.Module):
    """Pointwise heads *and* a pointwise trunk, so chunk and whole track agree exactly.

    Receptive field 1 in time everywhere: frame *t*'s outputs come from frame *t*'s
    features alone, which is what makes an exact comparison between a windowed scan and
    an uninterrupted one possible. A real TCN's edge frames genuinely differ.
    """

    HIDDEN = 6

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.onset = nn.Conv2d(1, 1, kernel_size=1)
        self.voicing = nn.Conv2d(1, 1, kernel_size=1)
        self.pitch = nn.Conv2d(1, 1, kernel_size=1)
        self.pool = nn.Conv2d(1, 1, kernel_size=(n_bins, 1))
        self.trunk = nn.Conv2d(1, self.HIDDEN, kernel_size=(n_bins, 1))

    def _scalar(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.pool(conv(x)).squeeze(1).squeeze(1).unsqueeze(-1)      # (B, T, 1)

    def heads_and_hidden(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        out = {
            "onset": self._scalar(self.onset, x),
            "voicing": self._scalar(self.voicing, x),
            "pitch": self.pitch(x).squeeze(1).transpose(1, 2),            # (B, T, F)
        }
        return out, self.trunk(x).squeeze(2).transpose(1, 2)              # (B, T, H)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads_and_hidden(x)[0]


def _decoder(transitions: bool = False) -> SemiCRFDecoder:
    # Seeded, because the two sides of the comparison build their own decoder and
    # this scorer -- unlike `LinearSpanScorer`, whose parameters are deterministic
    # constants -- is randomly initialised. Without this the test compares two
    # different models and reports the difference as a broken invariant.
    torch.manual_seed(1234)
    decoder = SemiCRFDecoder(
        stride=STRIDE,
        scorer=NeuralSpanScorer(
            LABELS, trunk_dim=_PointwiseTrunk.HIDDEN, frame_dim=4, order=2,
            span_dim=6, dur_dim=3, dmax_s=DMAX_S, dmax_frames=DMAX,
        ),
    )
    if transitions:
        # `NeuralSpanScorer` zero-inits `class_trans` and `interval`, so at rest every
        # transition scores 0 -- including `START -> y`. Any test that only ever runs
        # an untrained scorer therefore cannot tell `START` from a real predecessor,
        # which is exactly how the chunked gold score got away with assuming one.
        with torch.no_grad():
            g = torch.Generator().manual_seed(7)
            for p in (decoder.scorer.class_trans, decoder.scorer.interval):
                p.add_(torch.randn(p.shape, generator=g))
    return decoder


def _whole_track_nll(model, items, feats, transitions: bool = False) -> float:
    times = np.arange(N_CQT) * HOP_S
    decoder = _decoder(transitions)
    chunk = Chunk(
        features=feats[None],
        frame_times=times,
        gold=gold_segments(
            times[::STRIDE], items[0].reference_notes(), DMAX, decoder.scorer.labels
        ),
        y={}, stream=0, reset=True, prev=LABELS.START,
    )
    loss = SemiCRFLoss(decoder)
    batch = collate([chunk], context=0, right=0)
    with torch.no_grad():
        _, metrics = loss.compute(model, batch, torch.device("cpu"))
    return metrics["total"] * loss.scored_frames(batch, batch.features.shape[-1])


def _chunked_nll(model, cache, items, transitions: bool = False) -> float:
    dataset = ChunkDataset(
        train_items(items, CONDITIONS["clean"], cache),
        chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE, batch_size=1,
        right_context=0, labels=LABELS,
    )
    loss = SemiCRFLoss(_decoder(transitions))
    total = 0.0
    with torch.no_grad():
        for batch in dataset:
            _, metrics = loss.compute(model, batch, torch.device("cpu"))
            total += metrics["total"] * loss.scored_frames(
                batch, batch.features.shape[-1]
            )
    return total


@pytest.mark.parametrize("transitions", [False, True], ids=["zero-init", "trained"])
def test_the_neural_scorer_telescopes_across_chunks(track, transitions):  # noqa: F811
    """The suite's strongest invariant, applied to the scorer that has never seen it.

    Its potentials come from prefix sums over *absolute* frame index, and chunking moves
    that origin from the track's start to the window's. ``M_k`` is translation-invariant
    on paper -- ``(t - a)^k`` does not care where zero is -- but the implementation
    reconstructs it from ``P_p`` by binomial expansion in ``a``, which is where an origin
    could leak in without changing a single shape.

    **The ``trained`` arm is the one that found a bug**, and it is worth saying why the
    ``zero-init`` arm could not. ``segmentation_scores`` marked a row's first segment as
    following ``START``, which is true of a whole track and false of a chunk cut out of
    one -- there the predecessor is whatever the last chunk ended on, and ``log Z``
    applies exactly that. Two halves of the NLL, two different quantities. But an
    untrained ``NeuralSpanScorer`` has every transition at zero, so ``START -> y`` and
    ``note -> y`` are the same number and the identity holds anyway. Perturbing the
    transition table is the whole difference: 5.18 nats over these six chunks before the
    fix, ~0.9 per chunk boundary, in every run that ever trained a transition.
    """
    items, cache, feats = track
    model = _PointwiseTrunk()
    whole = _whole_track_nll(model, items, feats, transitions)
    chunked = _chunked_nll(model, cache, items, transitions)
    assert chunked == pytest.approx(whole, rel=1e-3, abs=1e-2), (
        f"chunked {chunked} != whole-track {whole}: the neural scorer's chunked "
        "objective is not the one it reports"
    )


def _second_track(tmp_path, cache) -> list[_Item]:
    """A track whose features differ, so a carry written from it is *detectably* wrong."""
    item = _Item("t1", tmp_path)
    Path(item.audio()).write_bytes(b"")
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((N_BINS, N_CQT)).astype(np.float32)
    CachedItem(item.name, feats, {}, np.arange(N_CQT) * HOP_S).save(
        cache.feature_path(item.audio())
    )
    return [item]


def _dataset(items, cache):
    return ChunkDataset(
        train_items(items, CONDITIONS["clean"], cache),
        chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE, batch_size=1,
        right_context=0, labels=LABELS,
    )


@pytest.mark.parametrize("scorer", ["linear", "neural"])
def test_a_validation_pass_between_epochs_leaves_training_unchanged(
    track, tmp_path, scorer  # noqa: F811
):
    """train -> val -> train, through one ``SemiCRFLoss``, must equal train -> train.

    Both loaders shard to ``base = 0`` under ``num_workers=0``, so they emit the *same*
    stream ids and the val pass writes its carries onto the train streams' keys. The
    dataset raises ``reset`` on each stream's first chunk of a fresh pass, which is
    supposed to make that harmless -- this is the test that says whether it does.

    Run for both scorers because a difference between them localises the fault: shared
    means the carry plumbing, neural-only means the new potentials.
    """
    items, cache, _ = track
    model = _PointwiseTrunk()
    train_ds = _dataset(items, cache)
    val_ds = _dataset(_second_track(tmp_path, cache), cache)
    decoder = (
        SemiCRFDecoder(stride=STRIDE) if scorer == "linear" else _decoder()
    )
    if scorer == "linear":
        # The binary alphabet's gold, matching the decoder's own labels.
        train_ds.labels = val_ds.labels = decoder.scorer.labels
    loss = SemiCRFLoss(decoder)

    def one_pass(dataset) -> list[float]:
        with torch.no_grad():
            return [
                loss.compute(model, b, torch.device("cpu"))[1]["total"] for b in dataset
            ]

    first = one_pass(train_ds)
    one_pass(val_ds)                      # the validation pass, same loss object
    third = one_pass(train_ds)
    assert third == pytest.approx(first, abs=1e-6), (
        "a validation pass changed the next training epoch's losses: the val loader's "
        "carries are landing on the train streams' keys"
    )
