"""The shared frame-tagging model: one CQT encoder, swappable per-frame heads.

Onset, voicing and pitch differ only in the head and the loss; the encoder --
convolutions over the (frequency, time) CQT, then a temporal stack -- is
identical, which is the whole reason they share a model. Convolving over
log-frequency lets a filter that recognises a pitch's harmonic stack apply at any
register (a fixed shift along the CQT axis).

Two temporal stacks are provided:

  - ``TCNEncoder`` (default) -- dilated 1-D convolutions over time. Parallel over
    the time axis, so fast on MPS and GPU alike, and its short-to-medium
    receptive field is a natural fit for the local attack shape onset detection
    keys on. This is the recommended encoder.
  - ``GruEncoder`` -- a BiGRU, kept only as an ablation baseline. Metal has no
    fused RNN kernel, so it is slow on MPS; use it to check the TCN loses nothing.

``FrameTagger`` composes an encoder with a ``ModuleDict`` of heads and returns a
dict of per-frame logits. Training a single head (the onset-first plan) is just a
one-entry dict. All shapes are (batch, time, ...) so a head is a plain per-frame
projection, and the two encoders are interchangeable.
"""

import torch
from torch import nn

from stemify.features import N_BINS


class _ConvFrontend(nn.Module):
    """CQT (B, 1, n_bins, T) -> per-frame channels (B, conv_channels * n_bins, T).

    Two conv blocks over (freq, time); padding keeps T, and frequency is kept
    (not pooled) so the temporal stack downstream sees the full harmonic layout.
    """

    def __init__(self, n_bins: int = N_BINS, conv_channels: int = 32):
        super().__init__()
        self.out_channels = conv_channels * n_bins
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)                       # (B, C, F, T)
        b, c, f, t = z.shape
        return z.reshape(b, c * f, t)          # (B, C*F, T)


class _TCNBlock(nn.Module):
    """Residual block of two dilated, non-causal (centred) 1-D convolutions."""

    def __init__(self, channels: int, dilation: int, kernel: int = 3):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # 'same' length for odd kernel
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual


class TCNEncoder(nn.Module):
    """CQT -> per-frame features via dilated temporal convolutions.

    ``levels`` residual blocks with dilations 1, 2, 4, ... give a receptive field
    that grows exponentially with depth: 4 levels of kernel-3 blocks see ~60
    frames (~0.7 s), ample for onset and voicing. All convolution, so it
    parallelises over time -- the reason to prefer it over the GRU on MPS.
    """

    def __init__(
        self,
        n_bins: int = N_BINS,
        conv_channels: int = 32,
        hidden: int = 128,
        levels: int = 4,
        kernel: int = 3,
    ):
        super().__init__()
        self.hidden = hidden
        self.frontend = _ConvFrontend(n_bins, conv_channels)
        self.project = nn.Conv1d(self.frontend.out_channels, hidden, kernel_size=1)
        self.tcn = nn.Sequential(
            *[_TCNBlock(hidden, dilation=2**i, kernel=kernel) for i in range(levels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frontend(x)          # (B, C*F, T)
        z = self.project(z)           # (B, hidden, T)
        z = self.tcn(z)               # (B, hidden, T)
        return z.transpose(1, 2)      # (B, T, hidden)


class GruEncoder(nn.Module):
    """CQT -> per-frame features via a BiGRU. Ablation baseline only (slow on MPS)."""

    def __init__(self, n_bins: int = N_BINS, conv_channels: int = 32, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.frontend = _ConvFrontend(n_bins, conv_channels)
        self.rnn = nn.GRU(
            input_size=self.frontend.out_channels,
            hidden_size=hidden // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frontend(x).transpose(1, 2)  # (B, T, C*F)
        out, _ = self.rnn(z)
        return out                             # (B, T, hidden)


class FrameHead(nn.Module):
    """A per-frame linear projection to ``out_dim`` logits (no activation)."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)  # (B, T, out_dim)


def onset_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: onset present. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)


def voicing_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: bass sounding. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)


def pitch_head(hidden: int = 128, n_classes: int = 45) -> FrameHead:
    """One logit per pitch class per frame (default 45 = MIDI 23-67, the 5-string
    range the confusion matrix already uses). Cross-entropy, or swap for the
    two-hot cents target later."""
    return FrameHead(hidden, n_classes)


class FrameTagger(nn.Module):
    """Encoder + named heads -> {head_name: (B, T, out_dim) logits}.

    ``heads`` may hold one entry (train onset alone) or several (joint). The
    encoder is shared across whatever heads are present, and any encoder exposing
    ``.hidden`` and returning (B, T, hidden) works -- TCN or GRU.
    """

    def __init__(self, encoder: nn.Module, heads: dict[str, FrameHead]):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {name: head(features) for name, head in self.heads.items()}