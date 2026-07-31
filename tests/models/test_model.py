"""Shape contracts for the neural model. No training -- just that each backbone
collapses frequency and preserves time, and that heads project per frame, so a
training loop can rely on (batch, time, out_dim). Both backbones are interchangeable.
"""

import pytest
import torch

from stemify.features import N_BINS
from stemify.models.heads import onset_head, pitch_head, voicing_head
from stemify.models.networks import GRU, TCN
from stemify.models.tagger import FrameTagger

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
    heads = {"onset": onset_head(net.hidden), "voicing": voicing_head(net.hidden)}
    pitch = pitch_head()
    model = FrameTagger(net, heads, frequency_heads={"pitch": pitch}).eval()
    out = model(_cqt_batch())
    assert out["onset"].shape == (B, T, 1)
    assert out["voicing"].shape == (B, T, 1)
    # Pitch is one logit per CQT bin, not per MIDI class: it is a position on the
    # frequency axis, and the two-hot target is expressed in those bins.
    assert out["pitch"].shape == (B, T, N_BINS)
    # The backbone is counted once, not duplicated per head: total params equal one
    # backbone plus the heads. A behavioural check of sharing -- no reach into the
    # tagger's internals.
    n_backbone = sum(p.numel() for p in net.parameters())
    n_heads = sum(p.numel() for h in heads.values() for p in h.parameters())
    n_pitch = sum(p.numel() for p in pitch.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    assert n_total == n_backbone + n_heads + n_pitch


def test_pitch_branch_is_frequency_equivariant():
    """The same harmonic pattern must score the same at any register.

    This is the property the flatten destroys and the whole reason the branch taps
    the front-end before it: shifting an input pattern up N bins must shift the
    logits up N bins, rather than landing on a differently-weighted position.
    """
    net, pitch = TCN().eval(), pitch_head().eval()
    x = torch.zeros(1, 1, N_BINS, T)
    x[0, 0, 60:66, :] = 5.0                    # a narrow "partial" at bin 60
    shifted = torch.zeros_like(x)
    shifted[0, 0, 90:96, :] = 5.0              # the same pattern, 30 bins up

    with torch.no_grad():
        a = pitch(net.frontend.conv_features(x))
        b = pitch(net.frontend.conv_features(shifted))
    # Compare away from the padded edges, where the shift is not well defined.
    assert torch.allclose(a[0, T // 2, 40:80], b[0, T // 2, 70:110], atol=1e-4)


# --------------------------------------------------------------------------- #
# input width: a mix-conditioned model reads several stacked renderings        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backbone_cls", [TCN, GRU])
def test_backbone_accepts_the_channel_count_it_was_built_for(backbone_cls):
    net = backbone_cls(in_channels=2).eval()
    out = net(torch.randn(B, 2, N_BINS, T))
    # Only the first conv widens: the temporal stack and every head are unchanged,
    # so the output contract a training loop relies on does not move.
    assert out.shape == (B, T, net.hidden)


def test_default_stays_single_channel():
    assert TCN().in_channels == 1
    assert FrameTagger(TCN(), {}).in_channels == 1


def test_tagger_reports_its_backbones_width():
    # Read off the backbone rather than stored, so it cannot drift from the conv
    # that actually has to accept the tensor.
    net = TCN(in_channels=2)
    assert FrameTagger(net, {"onset": onset_head(net.hidden)}).in_channels == 2


def test_a_wider_model_rejects_a_single_channel_batch():
    net = TCN(in_channels=2).eval()
    with pytest.raises(RuntimeError):
        net(_cqt_batch())
