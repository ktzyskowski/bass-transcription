"""Temporal backbones over the CQT front-end: ``TCN`` (default) and ``GRU``.

Both first apply the same conv front-end over the (frequency, time) CQT -- filters
that recognise a pitch's harmonic stack, applied at any register by convolving over
log-frequency -- then a temporal stack. The stack is what differs:

  - ``TCN`` (default) -- dilated 1-D convolutions over time. Parallel over the time
    axis, so fast on MPS and GPU alike, and its short-to-medium receptive field is a
    natural fit for the local attack shape onset detection keys on.
  - ``GRU`` -- a BiGRU, kept only as an ablation baseline. Metal has no fused RNN
    kernel, so it is slow on MPS; use it to check the TCN loses nothing.

Both expose ``.hidden`` and return (batch, time, hidden), so a ``FrameHead`` is a
plain per-frame projection and the two are interchangeable inside a ``FrameTagger``.
These are *backbones*, not the pipeline ``Encoder``: a ``NeuralEncoder`` composes one
of them with heads to produce ``Cues``.
"""

import torch
from torch import nn

from stemify.features import N_BINS


class _ConvFrontend(nn.Module):
    """CQT (B, in_channels, n_bins, T) -> per-frame channels (B, conv_channels * n_bins, T).

    Two conv blocks over (freq, time); padding keeps T, and frequency is kept
    (not pooled) so the temporal stack downstream sees the full harmonic layout.

    ``in_channels`` is how many renderings of the same music are stacked -- 1 for
    the bass alone, 2 to add the mixture it came from. Only the first conv's input
    width changes: the two CQTs share a frame grid and a frequency axis, so the
    first layer sees both registers of the same instant at the same bin and can
    learn "this transient is present in the mix too" as a plain filter, with
    everything downstream unchanged.
    """

    def __init__(
        self, n_bins: int = N_BINS, conv_channels: int = 32, in_channels: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = conv_channels * n_bins
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, F, T) -> (B, conv_channels, F, T), *before* the flatten.

        Split out because this is exactly what a pitch head has to consume: the
        flatten below gives every ``(channel, bin)`` pair its own downstream weight,
        which is what ends the frequency equivariance a pitch head depends on. See
        ``encoders/pitch_head.py``.
        """
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv_features(x)              # (B, C, F, T)
        b, c, f, t = z.shape
        return z.reshape(b, c * f, t)          # (B, C*F, T)


def tcn_receptive_field(levels: int = 4, kernel: int = 3) -> int:
    """How many input frames each output frame sees along time, for a ``TCN``.

    A free function as well as a property because callers need it *before* building a
    model: chunked training has to know how much audio to carry past a chunk's edge so
    the frames the DP will score are not computed against zero-padding, and it would
    rather not allocate a network to ask. Defaults match ``TCN``'s.

    Two 3x3 convs in the front-end span 5 frames; each residual block adds
    ``2 * (kernel - 1) * dilation`` with dilation doubling per level. At the defaults
    that is 5 + 4*(1+2+4+8) = 65 frames, ~0.75 s at the CQT hop.
    """
    return 5 + sum(2 * (kernel - 1) * 2**i for i in range(levels))


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


class TCN(nn.Module):
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
        in_channels: int = 1,
    ):
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.levels, self.kernel = levels, kernel
        self.frontend = _ConvFrontend(n_bins, conv_channels, in_channels)
        self.project = nn.Conv1d(self.frontend.out_channels, hidden, kernel_size=1)
        self.tcn = nn.Sequential(
            *[_TCNBlock(hidden, dilation=2**i, kernel=kernel) for i in range(levels)]
        )

    @property
    def receptive_field(self) -> int:
        return tcn_receptive_field(self.levels, self.kernel)

    def _temporal(self, flat: torch.Tensor) -> torch.Tensor:
        z = self.project(flat)        # (B, hidden, T)
        z = self.tcn(z)               # (B, hidden, T)
        return z.transpose(1, 2)      # (B, T, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._temporal(self.frontend(x))

    def spectral_and_temporal(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both altitudes from a single front-end pass: ``(B, C, F, T)`` for a pitch
        head, ``(B, T, hidden)`` for the per-frame heads.

        A separate method rather than a wider ``forward`` return, so the backbone
        contract every existing caller relies on -- and that ``GRU`` also satisfies --
        is unchanged. A backbone without this method simply cannot host a pitch head,
        which is the honest outcome: ``GRU`` is an ablation baseline, not a shape the
        frequency-resolved branch is defined for.
        """
        spectral = self.frontend.conv_features(x)   # (B, C, F, T)
        b, c, f, t = spectral.shape
        return spectral, self._temporal(spectral.reshape(b, c * f, t))


class GRU(nn.Module):
    """CQT -> per-frame features via a BiGRU. Ablation baseline only (slow on MPS)."""

    def __init__(
        self,
        n_bins: int = N_BINS,
        conv_channels: int = 32,
        hidden: int = 128,
        in_channels: int = 1,
    ):
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.frontend = _ConvFrontend(n_bins, conv_channels, in_channels)
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
