"""SemiCRFLoss under the generic Trainer: the structured objective end to end.

A tiny per-frame linear tagger over engineered features (a bin that marks onsets, a
bin that marks voiced frames) is trainable, so the generic Trainer + SemiCRFLoss must
drive the NLL down -- proof the structured path runs through the same loop as
FrameBCELoss, no audio and no real backbone.

Each track is fed as a *single* chunk (``context=0``, ``right=0``), which is the
degenerate case of the chunked loader and the closest thing left to the whole-track
path this file used to exercise.
"""

import numpy as np
import pytest
import torch
from torch import nn

from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY
from stemify.music.notes import NoteEvent
from stemify.fitting.config import TrainConfig
from stemify.semicrf.fit.geometry import ChunkGeometry
from stemify.semicrf.fit.dataset import Chunk, collate
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.training.trainer import Trainer

from spanscorer import CueScorer

DT = 0.0116
T = 240
TIMES = np.arange(T) * DT
N_BINS = 4
STRIDE = 3


class _LinearHeads(nn.Module):
    """Per-frame linear from features to onset/voicing logits -- trainable, batched."""

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.onset = nn.Linear(n_bins, 1)
        self.voicing = nn.Linear(n_bins, 1)

    def forward(self, x):  # (B, 1, n_bins, T)
        h = x.squeeze(1).transpose(1, 2)  # (B, T, n_bins)
        return {"onset": self.onset(h), "voicing": self.voicing(h)}  # each (B, T, 1)


def _chunk(spans, pitches_hz, stream: int) -> Chunk:
    """A track whose features *encode* the answer: bin 0 spikes at onsets, bin 1 is
    high on voiced frames -- so a linear head can learn to lower the NLL."""
    feats = np.zeros((N_BINS, T), dtype=np.float32)
    freqs = np.zeros(T, dtype=np.float32)
    for (a, b), hz in zip(spans, pitches_hz, strict=True):
        feats[0, round(a / DT)] = 1.0                       # onset marker
        feats[1, (TIMES >= a) & (TIMES < b)] = 1.0          # voiced marker
        freqs[(TIMES >= a) & (TIMES < b)] = hz
    notes = [NoteEvent(onset=a, offset=b, midi_pitch=45.0) for a, b in spans]
    # dmax from the decoder's own grid, so the gold and the scan agree on the bound.
    dmax = SemiCRFDecoder(CueScorer(), stride=STRIDE).scorer.dmax_for(TIMES[::STRIDE])
    return Chunk(
        # (1, n_bins, T): the channel axis is always present, single-channel or not.
        features=feats[None],
        frame_times=TIMES,
        gold=gold_segments(TIMES[::STRIDE], notes, dmax),
        y={},
        stream=stream,
        reset=True,
        prev=BINARY.START,
    )


def _batch():
    return collate(
        [
            _chunk([(0.2, 0.9), (1.2, 1.9)], [110, 146], stream=0),
            _chunk([(0.3, 0.8), (1.4, 2.0)], [98, 130], stream=1),
        ],
        ChunkGeometry(context=0, chunk=1, right=0, stride=STRIDE),
    )


def test_compute_returns_finite_loss_and_total_metric():
    loss = SemiCRFLoss(SemiCRFDecoder(CueScorer(), stride=STRIDE))
    total, metrics = loss.compute(_LinearHeads(), _batch(), torch.device("cpu"))
    assert torch.isfinite(total) and total.requires_grad
    assert metrics["total"] == metrics["nll"]
    assert len(_batch()) == 2   # the batch answers its own size now


def test_parameters_expose_the_decoder_potentials():
    decoder = SemiCRFDecoder(CueScorer(), stride=STRIDE)
    loss = SemiCRFLoss(decoder)
    assert set(loss.parameters()) == set(decoder.parameters())


def test_trainer_with_semicrf_loss_reduces_nll():
    model = _LinearHeads()
    loss = SemiCRFLoss(SemiCRFDecoder(CueScorer(), stride=STRIDE))
    config = TrainConfig(device="cpu", max_epochs=15, learning_rate=0.05, patience=99)
    # A plain list of batches is a valid loader -- the Trainer just iterates it.
    loader = [_batch() for _ in range(2)]
    history = Trainer(model, loss, config).fit(loader)
    assert history.train_loss[-1] < history.train_loss[0]


def test_a_checkpoint_of_a_retired_scorer_is_refused_by_name():
    # A checkpoint trained against the log-linear scorer records `kind: "linear"`, and
    # one older still records no scorer at all. Both describe a model that no longer
    # exists. Refusing them is right; the reason has to survive into the message, or
    # they read as corrupt files rather than superseded ones.
    from stemify.semicrf.fit.recipe import SemiCRFRecipe

    with pytest.raises(ValueError, match="unknown scorer kind 'linear'"):
        SemiCRFRecipe.from_config({"decoder": {"scorer": {"kind": "linear"}}})
    with pytest.raises(ValueError, match="records no scorer"):
        SemiCRFRecipe.from_config({"decoder": {"stride": 3, "dmax_s": 4.0}})
