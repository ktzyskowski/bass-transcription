"""The temporal backbone over the CQT front-end: ``TCN``.

A conv front-end over the (frequency, time) CQT -- filters that recognise a pitch's
harmonic stack, applied at any register by convolving over log-frequency -- then a
stack of dilated 1-D convolutions over time. Dilation is parallel over the time axis,
so it is fast on GPU, and its short-to-medium receptive field is a natural fit for the
local attack shape onset detection keys on.

There was also a ``GRU`` here, a BiGRU kept as an ablation baseline. It was removed
rather than left standing: nothing built one, and it could not host the frequency
head. ``CueNet`` taps ``frequency_and_frame`` for a ``FrequencyHead``, which only
this backbone defines -- so a tagger built on the GRU with a pitch head raised
``AttributeError`` on its first forward pass, while the module docstring advertised the
two as interchangeable. The path forward is a neural CRF, not a second backbone.

This is a *backbone*, not the pipeline ``Encoder``: a ``LearnedEncoder`` composes one
with heads to produce ``Frames``.
"""

from collections.abc import Sequence

import torch
from torch import nn

from stemify.audio.features import N_BINS, harmonic_shifts


class _ConvFrontend(nn.Module):
    """CQT (B, in_channels, n_bins, T) -> per-frame channels (B, conv_channels * bins, T).

    Two conv blocks over (freq, time); padding keeps T, and frequency is kept
    (not pooled) so the temporal stack downstream sees the full harmonic layout.

    ``in_channels`` is how many renderings of the same music are stacked -- 1 for
    the bass alone, 2 to add the mixture it came from. Only the first conv's input
    width changes: the two CQTs share a frame grid and a frequency axis, so the
    first layer sees both registers of the same instant at the same bin and can
    learn "this transient is present in the mix too" as a plain filter, with
    everything downstream unchanged.

    ``harmonics`` stacks each rendering against itself at its own harmonics -- see
    ``stack``. It multiplies the conv's input width the same way ``in_channels`` does
    and changes nothing else about this module.
    """

    def __init__(
        self,
        n_bins: int = N_BINS,
        conv_channels: int = 32,
        in_channels: int = 1,
        harmonics: Sequence[int] = (1,),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.harmonics = tuple(harmonics)
        self.shifts = harmonic_shifts(self.harmonics)
        # The fundamental axis loses whatever the topmost harmonic reaches past, so
        # every stacked channel is real data and none needs padding. Nothing is lost:
        # the bins given up sit above any bass fundamental (bin 144 is D#5), and their
        # content is still present -- as the harmonic channels of the notes below.
        self.n_bins = n_bins - max(self.shifts)
        self.out_channels = conv_channels * self.n_bins
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * len(self.shifts), conv_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, n_bins, T) -> (B, C * len(harmonics), self.n_bins, T).

        **The same numbers, re-indexed so a 3x3 kernel can reach them.** A note at bin
        ``b`` also radiates at ``b + 36``, ``b + 57``, ``b + 72``; a two-layer 3x3
        front-end sees +-2 bins, so as laid out it can never have a fundamental and any
        of its own harmonics inside one window, and the flatten then removes the
        frequency axis entirely. Slicing each harmonic to its own channel puts them at
        the same ``(bin, frame)`` position, where the existing kernel sees them together
        on layer one -- and learns *one* detector that works at every pitch instead of a
        separate one per bin.

        No information is added, only adjacency; that is also the argument for using a
        CQT over a linear spectrogram in the first place, one level down.

        Channel order is harmonic-major: all renderings at ``h=1``, then all at ``h=2``.
        The identity case (``harmonics=(1,)``) returns the input untouched rather than
        going through ``cat``, so the default path is exactly what it was before.
        """
        if self.shifts == (0,):
            return x
        return torch.cat([x[:, :, s: s + self.n_bins] for s in self.shifts], dim=1)

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, F, T) -> (B, conv_channels, self.n_bins, T), *before* the
        flatten.

        Split out because this is exactly what a pitch head has to consume: the
        flatten below gives every ``(channel, bin)`` pair its own downstream weight,
        which is what ends the frequency equivariance a pitch head depends on. See
        ``models/heads/pitch.py``.
        """
        # Annotated locals throughout this module: torch types `Module.__call__` as
        # returning `Any`, so a submodule call otherwise launders the tensor type away
        # and every function that returns one straight through declares more than the
        # checker can see.
        out: torch.Tensor = self.conv(self.stack(x))
        return out

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
    # ``1 << i`` rather than ``2**i``: the same dilation ladder (1, 2, 4, 8), but an
    # integer by construction. ``int ** int`` is not necessarily an int -- a negative
    # exponent gives a float -- so the checker widens it, and the width leaks out
    # through this function's declared ``int``.
    return 5 + sum(2 * (kernel - 1) * (1 << i) for i in range(levels))


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
        residual: torch.Tensor = self.net(x)
        return x + residual


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
        harmonics: Sequence[int] = (1,),
    ):
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.levels, self.kernel = levels, kernel
        self.frontend = _ConvFrontend(n_bins, conv_channels, in_channels, harmonics)
        self.project = nn.Conv1d(self.frontend.out_channels, hidden, kernel_size=1)
        self.tcn = nn.Sequential(
            *[_TCNBlock(hidden, dilation=2**i, kernel=kernel) for i in range(levels)]
        )

    @property
    def receptive_field(self) -> int:
        return tcn_receptive_field(self.levels, self.kernel)

    def _temporal(self, flat: torch.Tensor) -> torch.Tensor:
        z: torch.Tensor = self.project(flat)   # (B, hidden, T)
        z = self.tcn(z)                        # (B, hidden, T)
        return z.transpose(1, 2)               # (B, T, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._temporal(self.frontend(x))

    def frequency_and_frame(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both head inputs from a single front-end pass, in dict order:
        ``(B, C, F, T)`` for a ``FrequencyHead``, ``(B, T, hidden)`` for a ``FrameHead``.

        A separate method rather than a wider ``forward`` return: a ``FrameHead`` needs
        only ``(B, T, hidden)``, and widening ``forward`` would make every caller carry
        a tensor most of them discard.
        """
        frequency = self.frontend.conv_features(x)   # (B, C, F, T)
        b, c, f, t = frequency.shape
        return frequency, self._temporal(frequency.reshape(b, c * f, t))
