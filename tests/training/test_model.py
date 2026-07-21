"""Shape contracts for the model. No training -- just that the encoder collapses
frequency and preserves time, and that heads project per frame, so a training
loop can rely on (batch, time, out_dim) everywhere.
"""

import torch

from stemify.features import N_BINS
from stemify.training.model import (
    FrameEncoder,
    FrameTagger,
    onset_head,
    pitch_head,
    voicing_head,
)

B, T = 2, 20


def _cqt_batch() -> torch.Tensor:
    return torch.randn(B, 1, N_BINS, T)


def test_encoder_collapses_freq_keeps_time():
    enc = FrameEncoder().eval()
    out = enc(_cqt_batch())
    assert out.shape == (B, T, enc.hidden)


def test_single_head_tagger():
    enc = FrameEncoder().eval()
    model = FrameTagger(enc, {"onset": onset_head(enc.hidden)}).eval()
    out = model(_cqt_batch())
    assert set(out) == {"onset"}
    assert out["onset"].shape == (B, T, 1)


def test_multi_head_shares_encoder():
    enc = FrameEncoder().eval()
    model = FrameTagger(
        enc,
        {
            "onset": onset_head(enc.hidden),
            "voicing": voicing_head(enc.hidden),
            "pitch": pitch_head(enc.hidden, n_classes=45),
        },
    ).eval()
    out = model(_cqt_batch())
    assert out["onset"].shape == (B, T, 1)
    assert out["voicing"].shape == (B, T, 1)
    assert out["pitch"].shape == (B, T, 45)
    # one shared encoder, not one per head
    assert sum(1 for _ in model.encoder.parameters()) > 0