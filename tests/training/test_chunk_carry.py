"""``SemiCRFLoss``'s carry: the property the whole chunked design rests on.

Summing the per-chunk losses over a track must reproduce the NLL of one uninterrupted
scan over that same track. If it does not, chunked training optimises something other
than what it reports, and every number a step-matched run produces is uninterpretable.

The reference here is a *single chunk spanning the whole track* (``context=0``,
``right=0``, one scan, no carry), which is what the deleted whole-track loss computed.
The DP half is covered independently in ``tests/decoders/test_semicrf.py``, against
brute-force enumeration of every segmentation -- that is the check that grounds this
one, and it does not compare the implementation to itself. What is tested *here* is the
layer above: that the loss drives the decoder correctly across a loader -- resuming the
right carry per stream, resetting at track boundaries, and subtracting the right gold
pieces.

The model is deliberately **pointwise** (receptive field 1). A real TCN sees 65 frames,
so its cues near a chunk edge are computed partly from the window's zero-padding and
genuinely differ from the whole-track ones; that is what ``right_context`` exists to
shrink, and it is not what this test is about. With a pointwise model the two paths get
identical cues at identical frames, so any discrepancy is the loss's own.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stemify.cache import Cache
from stemify.contour import Contour
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.decoders.semicrf.gold import gold_segments
from stemify.notes import NoteEvent
from stemify.serde import TrackExample
from stemify.sources import Sources
from stemify.training.conditions import CONDITIONS
from stemify.training.semicrf.dataset import Chunk, ChunkDataset, collate
from stemify.training.semicrf.loss import SemiCRFLoss

N_BINS, HOP_S, STRIDE = 12, 0.0116, 3
CHUNK, DMAX = 10, 30
N_DP = CHUNK * 6                    # an exact number of chunks: no ragged tail dropped
N_CQT = N_DP * STRIDE


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
        return [
            NoteEvent(onset=t, offset=t + 0.25, midi_pitch=40.0)
            for t in np.arange(0.05, span - 0.4, 0.35)
        ]


class _NoOpCache(Cache):
    """The fixture writes its CQT directly; pyin never runs on the empty audio."""

    def ensure_features(self, dataset, transform=None, role="bass"):
        return [self.feature_path(i.sources().role(role)) for i in dataset]

    def contour(self, estimator, audio):
        return Contour(
            times=np.arange(N_CQT) * HOP_S, freqs=np.full(N_CQT, 110.0, dtype=np.float32)
        )


class _Pointwise(nn.Module):
    """Frame *t*'s logits from frame *t*'s features alone -- receptive field 1.

    So the cues a chunk produces are bit-identical to the whole track's at the same
    absolute frames, which is what makes an exact comparison possible at all.
    """

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.onset = nn.Conv2d(1, 1, kernel_size=1)
        self.voicing = nn.Conv2d(1, 1, kernel_size=1)
        self.pool = nn.Conv2d(1, 1, kernel_size=(n_bins, 1))  # collapse frequency

    def _head(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.pool(conv(x)).squeeze(1).squeeze(1).unsqueeze(-1)  # (B, T, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"onset": self._head(self.onset, x), "voicing": self._head(self.voicing, x)}


@pytest.fixture
def track(tmp_path):
    cache = _NoOpCache(tmp_path / "cache")
    item = _Item("t0", tmp_path)
    Path(item.audio()).write_bytes(b"")
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((N_BINS, N_CQT)).astype(np.float32)
    TrackExample(item.name, feats, {}, np.arange(N_CQT) * HOP_S).save(
        cache.feature_path(item.audio())
    )
    return [item], cache, feats


def _whole_track_nll(model, cache, items, feats) -> float:
    """The reference: the whole track as ONE chunk, so the scan runs uninterrupted.

    ``context=0``/``right=0`` and a fresh carry make this exactly ``log Z - gold`` over
    the track -- what the removed whole-track loss computed, with none of its padding.
    """
    times = np.arange(N_CQT) * HOP_S
    decoder = SemiCRFDecoder(stride=STRIDE)
    dmax = decoder.scorer.grid(times[::STRIDE]).dmax
    chunk = Chunk(
        features=feats[None],                       # (1, n_bins, T)
        frame_times=times,
        gold=gold_segments(times[::STRIDE], items[0].reference_notes(), dmax),
        y={},
        stream=0,
        reset=True,
    )
    loss = SemiCRFLoss(decoder)
    with torch.no_grad():
        _, metrics = loss.compute(
            model, collate([chunk], context=0, right=0), torch.device("cpu")
        )
    return metrics["total"]


def _chunked_nll(model, cache, items, right: int) -> float:
    """The sum of per-chunk telescoping losses over the same track."""
    dataset = ChunkDataset(
        items, cache, CONDITIONS["clean"], chunk_frames=CHUNK, dmax=DMAX,
        stride=STRIDE, batch_size=1, right_context=right,
    )
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE))
    total = 0.0
    with torch.no_grad():
        for batch in dataset:
            _, metrics = loss.compute(model, batch, torch.device("cpu"))
            total += metrics["total"]
    return total


@pytest.mark.parametrize("right", [0, 4])
def test_chunk_losses_sum_to_the_whole_track_nll(track, right):
    items, cache, feats = track
    model = _Pointwise()
    whole = _whole_track_nll(model, cache, items, feats)
    chunked = _chunked_nll(model, cache, items, right)
    assert chunked == pytest.approx(whole, abs=1e-2), (
        f"chunked {chunked} != whole-track {whole}; the telescoping loss is not the "
        "objective it claims to be"
    )


def test_a_second_epoch_gives_the_same_losses(track):
    # The carry is state held across `compute` calls, so a stale entry would leak from
    # one pass into the next. The dataset raises `reset` on each track's first chunk,
    # which must be enough to make a fresh pass identical -- otherwise epoch 2 resumes
    # from epoch 1's alpha and every epoch after the first is quietly wrong.
    items, cache, _ = track
    model = _Pointwise()
    dataset = ChunkDataset(
        items, cache, CONDITIONS["clean"], chunk_frames=CHUNK, dmax=DMAX,
        stride=STRIDE, batch_size=1, right_context=0,
    )
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE))

    def one_pass():
        with torch.no_grad():
            return [
                loss.compute(model, b, torch.device("cpu"))[1]["total"] for b in dataset
            ]

    assert one_pass() == pytest.approx(one_pass(), abs=1e-6)


def test_continuing_a_stream_without_a_carry_is_rejected(track):
    # Silent-failure guard: if a loader ever emitted a stream's chunks out of order, or
    # dropped the reset flag, the loss would otherwise start mid-track from a fresh
    # window and report a plausible number for the wrong quantity.
    items, cache, _ = track
    model = _Pointwise()
    dataset = ChunkDataset(
        items, cache, CONDITIONS["clean"], chunk_frames=CHUNK, dmax=DMAX,
        stride=STRIDE, batch_size=1, right_context=0,
    )
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE))
    batches = list(dataset)
    assert batches[1].resets == [False], "the fixture must have a continuing chunk"
    with pytest.raises(KeyError, match="no carry"):
        loss.compute(model, batches[1], torch.device("cpu"))


def test_epoch_batches_survives_a_streaming_loader(track):
    # The chunked loader has no len(); MLflow logs train_batches as a param, and
    # assuming every loader is sized crashed a run at the first write while every
    # smoke test passed (they ran without a tracking URI, so never logged params).
    from torch.utils.data import DataLoader

    from stemify.training.train import _epoch_batches

    items, cache, _ = track
    streaming = DataLoader(
        ChunkDataset(
            items, cache, CONDITIONS["clean"], chunk_frames=CHUNK, dmax=DMAX,
            stride=STRIDE, batch_size=1,
        ),
        batch_size=None,
    )
    assert _epoch_batches(streaming) == "streaming"
    assert _epoch_batches(DataLoader([1, 2, 3], batch_size=1)) == 3
