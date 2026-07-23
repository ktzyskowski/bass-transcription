"""Shape contracts for the neural model. No training -- just that each backbone
collapses frequency and preserves time, and that heads project per frame, so a
training loop can rely on (batch, time, out_dim). Both backbones are interchangeable.
"""

import pytest
import torch

from stemify.encoders.heads import onset_head, pitch_head, voicing_head
from stemify.encoders.networks import GRU, TCN
from stemify.encoders.tagger import FrameTagger
from stemify.features import N_BINS

B, T = 2, 20


def _cqt_batch() -> torch.Tensor:
    return torch.randn(B, 1, N_BINS, T)


@pytest.mark.parametrize("backbone_cls", [TCN, GRU])
def test_backbone_collapses_freq_keeps_time(backbone_cls):
    net = backbone_cls().eval()
    out = net(_cqt_batch())
    assert out.shape == (B, T, net.hidden)


def test_tcn_is_all_convolution():
    # No recurrent modules -- the property that makes it parallel over time.
    net = TCN()
    assert not any(isinstance(m, torch.nn.RNNBase) for m in net.modules())


def test_single_head_tagger():
    net = TCN().eval()
    model = FrameTagger(net, {"onset": onset_head(net.hidden)}).eval()
    out = model(_cqt_batch())
    assert set(out) == {"onset"}
    assert out["onset"].shape == (B, T, 1)


def test_multi_head_shares_one_backbone():
    net = TCN()
    heads = {
        "onset": onset_head(net.hidden),
        "voicing": voicing_head(net.hidden),
        "pitch": pitch_head(net.hidden, n_classes=45),
    }
    model = FrameTagger(net, heads).eval()
    out = model(_cqt_batch())
    assert out["onset"].shape == (B, T, 1)
    assert out["voicing"].shape == (B, T, 1)
    assert out["pitch"].shape == (B, T, 45)
    # The backbone is counted once, not duplicated per head: total params equal one
    # backbone plus the heads. A behavioural check of sharing -- no reach into the
    # tagger's internals.
    n_backbone = sum(p.numel() for p in net.parameters())
    n_heads = sum(p.numel() for h in heads.values() for p in h.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    assert n_total == n_backbone + n_heads
