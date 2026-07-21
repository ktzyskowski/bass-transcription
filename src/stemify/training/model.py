"""The shared frame-tagging model: one CQT encoder, swappable per-frame heads.

Onset, voicing and pitch differ only in the head and the loss; the encoder --
convolutions over the (frequency, time) CQT, then a recurrent layer for temporal
context -- is identical, which is the whole reason they share a model. Convolving
over log-frequency lets a filter that recognises a pitch's harmonic stack apply
at any register (a fixed shift along the CQT axis); the recurrent layer gives the
onset and voicing heads the short-horizon context a single frame lacks.

``FrameTagger`` composes the encoder with a ``ModuleDict`` of heads and returns a
dict of per-frame logits. Training a single head (the onset-first plan) is just a
one-entry dict -- no multi-task machinery forced on a task that does not want it.
All shapes are (batch, time, ...) so a head is a plain per-frame projection.
"""

import torch
from torch import nn

from stemify.features import N_BINS


class FrameEncoder(nn.Module):
    """CQT (B, 1, n_bins, T) -> per-frame features (B, T, hidden)."""

    def __init__(self, n_bins: int = N_BINS, conv_channels: int = 32, hidden: int = 128):
        super().__init__()
        # Two conv blocks over (freq, time). Padding keeps T; freq is pooled away
        # so the recurrent layer sees one vector per frame.
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )
        self.hidden = hidden
        # BiGRU: a frame's onset/voicing depends on what comes just before and
        # after, so context flows both directions.
        self.rnn = nn.GRU(
            input_size=conv_channels * n_bins,
            hidden_size=hidden // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T) -> conv -> (B, C, F, T)
        z = self.conv(x)
        b, c, f, t = z.shape
        # collapse channels+freq into one per-frame feature vector, time-major
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.rnn(z)
        return out  # (B, T, hidden)


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
    encoder is shared across whatever heads are present.
    """

    def __init__(self, encoder: FrameEncoder, heads: dict[str, FrameHead]):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {name: head(features) for name, head in self.heads.items()}