"""SemiCRFLoss under the generic Trainer: the structured objective end to end.

A tiny per-frame linear tagger over engineered features (a bin that marks onsets, a
bin that marks voiced frames) is trainable, so the generic Trainer + SemiCRFLoss must
drive the NLL down -- proof the whole-track structured path runs through the same
loop as FrameBCELoss, no audio and no real backbone.
"""

import numpy as np
import torch
from torch import nn

from stemify.contour import Contour
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.notes import NoteEvent
from stemify.training.config import TrainConfig
from stemify.training.semicrf.dataset import SequenceExample, collate
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.trainer import Trainer

DT = 0.0116
T = 240
TIMES = np.arange(T) * DT
N_BINS = 4


class _LinearHeads(nn.Module):
    """Per-frame linear from features to onset/voicing logits -- trainable, batched."""

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.onset = nn.Linear(n_bins, 1)
        self.voicing = nn.Linear(n_bins, 1)

    def forward(self, x):  # (B, 1, n_bins, T)
        h = x.squeeze(1).transpose(1, 2)  # (B, T, n_bins)
        return {"onset": self.onset(h), "voicing": self.voicing(h)}  # each (B, T, 1)


def _example(spans, pitches_hz):
    """A track whose features *encode* the answer: bin 0 spikes at onsets, bin 1 is
    high on voiced frames -- so a linear head can learn to lower the NLL."""
    feats = np.zeros((N_BINS, T), dtype=np.float32)
    freqs = np.zeros(T, dtype=np.float32)
    for (a, b), hz in zip(spans, pitches_hz):
        feats[0, round(a / DT)] = 1.0                       # onset marker
        feats[1, (TIMES >= a) & (TIMES < b)] = 1.0          # voiced marker
        freqs[(TIMES >= a) & (TIMES < b)] = hz
    notes = [NoteEvent(onset=a, offset=b, midi_pitch=45.0) for a, b in spans]
    # (1, n_bins, T): the channel axis is always present, single-channel or not.
    return SequenceExample(feats[None], TIMES, Contour(times=TIMES, freqs=freqs), notes)


def _batch():
    return collate([
        _example([(0.2, 0.9), (1.2, 1.9)], [110, 146]),
        _example([(0.3, 0.8), (1.4, 2.0)], [98, 130]),
    ])


def test_compute_returns_finite_loss_and_total_metric():
    loss = SemiCRFLoss(SemiCRFDecoder())
    total, metrics = loss.compute(_LinearHeads(), _batch(), torch.device("cpu"))
    assert torch.isfinite(total) and total.requires_grad
    assert metrics["total"] == metrics["nll"]
    assert loss.batch_size(_batch()) == 2


def test_parameters_expose_the_decoder_potentials():
    decoder = SemiCRFDecoder()
    loss = SemiCRFLoss(decoder)
    assert set(loss.parameters()) == set(decoder.parameters())


def test_trainer_with_semicrf_loss_reduces_nll():
    model = _LinearHeads()
    loss = SemiCRFLoss(SemiCRFDecoder())
    config = TrainConfig(device="cpu", max_epochs=15, learning_rate=0.05, patience=99)
    # A plain list of batches is a valid loader -- the Trainer just iterates it.
    loader = [_batch() for _ in range(2)]
    history = Trainer(model, loss, config).fit(loader)
    assert history.train_loss[-1] < history.train_loss[0]
