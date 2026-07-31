"""The learned pitch head: a frequency-resolved branch, and its two-hot bin code.

Onset and voicing are per-frame *scalars*, so their heads are a linear projection of
the backbone's ``(B, T, hidden)`` features. Pitch is not: it is a position along the
frequency axis, and by the time the backbone has produced those features that axis is
gone. ``_ConvFrontend`` flattens ``(B, C, F, T) -> (B, C*F, T)`` and projects with a
1x1 conv, which gives every ``(channel, bin)`` pair its own weight -- ~884k of the
TCN's 1.29M parameters, and the end of any frequency equivariance. A filter that
learned to recognise a harmonic stack at E1 has to learn it again at E2.

So the pitch branch taps the front-end *before* that flatten and stays 2-D
throughout, ending in a 1x1 conv to a single channel: ``(B, C, F, T) -> (B, T, F)``,
one logit per CQT bin. Every weight is shared across frequency, so the harmonic
pattern is learned once and applies at every register, and the output is already
aligned to the bins the target is expressed in.

**Why two-hot.** The CQT has 3 bins per semitone (~33 cents), which is coarse next to
the 50-cent matching tolerance -- rounding to the nearest bin would spend most of the
budget before the model made an error. Splitting each target across its two
neighbouring bins in proportion to where it falls between them makes the code
*linearly decodable*: the weighted centroid of the two recovers the exact fractional
bin, so resolution is limited by the head's confidence rather than by the bin width.
It also leaves the head free to put mass in two distant places at once -- an octave
ambiguity reads as two peaks rather than one confident average of the two, which is
the failure mode a single regressed pitch value cannot even express.
"""

import numpy as np
import torch
from torch import nn

from stemify.features import BINS_PER_OCTAVE, FMIN, N_BINS, midi_to_bin

# Frequency dilations grow ~3x per level so the branch can see across harmonics
# rather than only its immediate neighbours: 1 + 2*(1+3+9+27) = 81 bins, about 2.25
# octaves, which reaches a low bass fundamental's first few partials. Time dilations
# double as in the TCN. Both are small -- 4 blocks of 32 channels is ~37k parameters,
# against the 884k the flatten spends to destroy this structure.
FREQ_DILATIONS = (1, 3, 9, 27)
TIME_DILATIONS = (1, 2, 4, 8)


class PitchBranch(nn.Module):
    """Front-end features ``(B, C, F, T)`` -> per-bin pitch logits ``(B, T, F)``."""

    def __init__(self, conv_channels: int = 32, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.n_bins = n_bins
        blocks: list[nn.Module] = []
        for fd, td in zip(FREQ_DILATIONS, TIME_DILATIONS):
            blocks += [
                nn.Conv2d(
                    conv_channels, conv_channels, kernel_size=3,
                    padding=(fd, td), dilation=(fd, td),
                ),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU(),
            ]
        self.net = nn.Sequential(*blocks)
        # 1x1 to a single channel: the frequency axis becomes the output axis, so
        # nothing here is position-specific along it.
        self.out = nn.Conv2d(conv_channels, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.out(self.net(z))            # (B, 1, F, T)
        return h.squeeze(1).transpose(1, 2)  # (B, T, F)


def two_hot(midi: torch.Tensor, n_bins: int = N_BINS) -> torch.Tensor:
    """Per-frame MIDI pitch -> a two-hot distribution over CQT bins.

    ``midi`` is ``(...)`` with NaN at unvoiced frames; the result is ``(..., n_bins)``
    and is **all zeros** wherever the input was NaN, so an unvoiced frame carries no
    pitch evidence. Callers mask those frames out of the loss rather than training the
    head to predict "no pitch" -- that is the voicing head's job, and asking this one
    to do it too would put the two in competition.

    Mass is split between the two bins the true pitch falls between, in proportion to
    its distance from each, so ``decode_bins`` inverts it exactly.
    """
    voiced = ~torch.isnan(midi)
    safe = torch.where(voiced, midi, torch.zeros_like(midi))
    # midi_to_bin is a plain log-ratio, so it applies elementwise to a tensor.
    bins = BINS_PER_OCTAVE * torch.log2(
        (440.0 * 2.0 ** ((safe - 69.0) / 12.0)) / FMIN
    )
    bins = bins.clamp(0, n_bins - 1)
    lo = bins.floor()
    frac = bins - lo
    lo_idx = lo.long().clamp(0, n_bins - 1)
    hi_idx = (lo_idx + 1).clamp(0, n_bins - 1)

    target = torch.zeros(*midi.shape, n_bins, device=midi.device, dtype=torch.float32)
    target.scatter_(-1, lo_idx.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    # `add` rather than assign: at the top bin lo_idx == hi_idx and the two shares
    # must sum to one there rather than the second overwriting the first.
    target.scatter_add_(-1, hi_idx.unsqueeze(-1), frac.unsqueeze(-1))
    return target * voiced.unsqueeze(-1)


def decode_bins(logits: np.ndarray, neighbourhood: int = 1) -> np.ndarray:
    """Per-bin logits ``(T, n_bins)`` -> frequency in Hz per frame ``(T,)``.

    Takes the argmax bin and refines it with the probability-weighted centroid of its
    immediate neighbours -- the inverse of ``two_hot`` when the head is confident, and
    a graceful degradation to "near the peak" when it is not. Deliberately *local*: a
    centroid over the whole axis would place an octave-ambiguous frame halfway between
    the two candidates, at a pitch the model never proposed.
    """
    if logits.size == 0:
        return np.zeros(0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    peak = probs.argmax(axis=-1)

    offsets = np.arange(-neighbourhood, neighbourhood + 1)
    idx = np.clip(peak[:, None] + offsets[None, :], 0, logits.shape[-1] - 1)
    local = np.take_along_axis(probs, idx, axis=-1)
    total = local.sum(axis=-1)
    # A frame with no mass anywhere would divide by zero; it has no pitch to report.
    centroid = np.where(total > 0, (local * idx).sum(axis=-1) / np.maximum(total, 1e-9), peak)
    return FMIN * 2.0 ** (centroid / BINS_PER_OCTAVE)


def bin_of(midi: float) -> float:
    """The (fractional) CQT bin a MIDI pitch lands on -- the scalar form of the above."""
    return float(midi_to_bin(midi))
