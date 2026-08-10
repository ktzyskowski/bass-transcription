"""The learned pitch head: a frequency-resolved branch, and its two-hot bin code.

Onset and voicing are per-frame *scalars*, so their heads are a linear projection of
the backbone's ``(B, T, hidden)`` features. Pitch is not: it is a position along the
frequency axis, and by the time the backbone has produced those features that axis is
gone. ``_ConvFrontend`` flattens ``(B, C, F, T) -> (B, C*F, T)`` and projects with a
1x1 conv, which gives every ``(channel, bin)`` pair its own weight -- the bulk of the
TCN's parameters, and the end of any frequency equivariance. A filter that learned to
recognise a harmonic stack at E1 has to learn it again at E2.

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
from torch.nn import functional as F

from stemify.core.contour import Contour
from stemify.audio.features import BINS_PER_OCTAVE, FMIN, N_BINS, midi_to_bin

# Frequency dilations grow ~3x per level so the branch can see across harmonics rather
# than only its immediate neighbours: 1 + 2*(1+3+9+27) = 81 bins, about 2.25 octaves,
# which reaches a low bass fundamental's first few partials. Time dilations double as in
# the TCN. Both stacks are deliberately narrow -- sharing weights across frequency buys
# the receptive field for a small fraction of what the flatten spends destroying it.
FREQ_DILATIONS = (1, 3, 9, 27)
TIME_DILATIONS = (1, 2, 4, 8)


class FrequencyHead(nn.Module):
    """Front-end features ``(B, C, F, T)`` -> one logit per frequency bin ``(B, T, F)``.

    The peer of ``FrameHead``, named for the same thing it is: the granularity of what
    comes *out*. A ``FrameHead`` emits one value per frame; this emits one per frequency
    bin per frame. That difference is why the two tap the backbone at different points
    and why ``CueNet`` keeps them in separate dicts.

    Pitch is its only instance today. It is written as the general shape rather than as
    "the pitch branch" because nothing in it is about pitch -- a polyphonic pitch head,
    or a string/fret head over the same axis, is this class with a different target.
    """

    def __init__(self, conv_channels: int = 32, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.n_bins = n_bins
        blocks: list[nn.Module] = []
        for fd, td in zip(FREQ_DILATIONS, TIME_DILATIONS, strict=True):
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
        h: torch.Tensor = self.out(self.net(z))   # (B, 1, F, T)
        return h.squeeze(1).transpose(1, 2)       # (B, T, F)


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


def pitch_ce(logits: torch.Tensor, midi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the CQT bin axis against the two-hot code, voiced frames only.

    ``logits`` is ``(B, T, n_bins)``; ``midi`` is ``(B, T)`` continuous MIDI with NaN
    where the bass is silent.

    Lives beside the head rather than in a strategy package. It was in
    ``training.frame.loss``, which made it unreachable from the semi-CRF's auxiliary
    term without one strategy importing another's internals. This module already owns
    the head's encoding (``two_hot``, which this calls), its decoding (``decode_bins``)
    and its construction (``pitch_head``); its loss is the fourth thing about the same
    head, and it is a pure function of logits and targets, so nothing about ``models``
    depending on nothing is given up.

    **Why cross-entropy and not per-bin BCE.** The two-hot target is a proper
    distribution -- its rows sum to 1 -- so the axis is a choice among bins, not
    ``n_bins`` independent yes/no questions. Treating it as the latter is not merely a
    different parameterisation, it puts the term on the wrong *scale*: 216 independent
    sigmoids start near ``216 * ln 2 ~ 150``, against ``ln 216 ~ 5.4`` for a choice
    among 216 options. Summed BCE therefore arrives ~400x the binary heads on a shared
    trunk, which no loss weight should have to correct. This is the same reason
    DreamerV3 scores its two-hot critic and reward heads with categorical cross-entropy:
    get the functional form right and fixed coefficients work across scales.

    Unvoiced frames are dropped rather than trained toward some "no pitch" target.
    Where the bass sounds is the voicing head's question, and supervising this head on
    it too would have the two compete over a shared trunk to answer the same thing.
    """
    voiced = ~torch.isnan(midi)
    if not bool(voiced.any()):
        # Nothing to learn from this batch, but the term must stay in the graph or the
        # optimiser sees a different set of parameters depending on the batch drawn.
        return logits.sum() * 0.0
    # Soft-target cross-entropy: -sum_b target[b] * log q[b]. F.cross_entropy accepts
    # probability targets too, but only with the class axis at dim 1 -- spelling it out
    # keeps the bin axis last, matching two_hot and the head's own output.
    # The target's width comes from the *logits*, not from `N_BINS`. A harmonic-stacked
    # front-end crops the fundamental axis (see `_ConvFrontend.stack`), so the head emits
    # fewer bins than the CQT has; a target built at the module default is simply the
    # wrong shape against it, and the multiply below raises rather than trains.
    per_frame = -(two_hot(midi, logits.shape[-1]) * F.log_softmax(logits, dim=-1)).sum(-1)
    return (per_frame * voiced).sum() / voiced.sum()


def decode_bins(logits: np.ndarray, neighbourhood: int = 1) -> np.ndarray:
    """Per-bin logits ``(T, n_bins)`` -> frequency in Hz per frame ``(T,)``.

    Takes the argmax bin and refines it with the probability-weighted centroid of its
    immediate neighbours -- the inverse of ``two_hot`` when the head is confident, and
    a graceful degradation to "near the peak" when it is not. Deliberately *local*: a
    centroid over the whole axis would place an octave-ambiguous frame halfway between
    the two candidates, at a pitch the model never proposed.

    Softmax, not per-bin sigmoid: it has to invert the distribution the head is
    *trained* under, and that is a choice among bins (see
    ``pitch_ce`` below).
    """
    if logits.size == 0:
        return np.zeros(0)
    shifted = logits - logits.max(axis=-1, keepdims=True)   # stable softmax
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    peak = probs.argmax(axis=-1)

    offsets = np.arange(-neighbourhood, neighbourhood + 1)
    idx = np.clip(peak[:, None] + offsets[None, :], 0, logits.shape[-1] - 1)
    local = np.take_along_axis(probs, idx, axis=-1)
    total = local.sum(axis=-1)
    # A frame with no mass anywhere would divide by zero; it has no pitch to report.
    centroid = np.where(total > 0, (local * idx).sum(axis=-1) / np.maximum(total, 1e-9), peak)
    hz: np.ndarray = FMIN * 2.0 ** (centroid / BINS_PER_OCTAVE)
    return hz


# Above this, the voicing head is confident enough that the pitch head's guess for a
# frame is worth reporting at all. Lives beside ``decode_contour`` because it is part of
# reading this head's output, and both callers of that function need it: LearnedEncoder,
# which has the logits already, and NeuralPitch, which computes them from a path.
#
# Deliberately *not* the decoder's ``VOICING_THRESHOLD``. The two look like the same
# number and answer different questions -- this asks "does this frame have a pitch worth
# carrying", the decoder's asks "is a note sounding here" -- and they sweep to different
# optima. While they shared a constant, retuning the decoder silently retuned the
# contour gate too.
CONTOUR_VOICING_THRESHOLD = 0.5


def decode_contour(
    logits: np.ndarray,
    frame_times: np.ndarray,
    voiced: np.ndarray | None = None,
) -> Contour:
    """Per-bin logits ``(T, n_bins)`` -> a ``Contour`` on the feature grid.

    The single place the head's output becomes a pitch track, so the two callers that
    need it -- ``LearnedEncoder`` (which already has the logits) and ``NeuralPitch``
    (which computes them from a path) -- cannot drift into two different decodes.

    Two things fall out that pyin could not give us. The contour shares the CQT's
    ~11.6 ms grid rather than pyin's 23.2 ms one, so a note boundary is no longer
    quantised to half the match tolerance -- the quantisation behind the ``MIN_NOTE_S``
    interaction that made the pitch-split cue a liability. And ``voiced`` marks the
    rests, so "there is no pitch here" and "the bass is silent here" stop being separate
    claims made by two components that can disagree.

    ``voiced`` is a boolean mask, not a curve and a threshold: the threshold belongs to
    whoever owns the decoder's operating point, and importing it here would point
    ``models`` at ``decoders``. With no mask every frame carries the head's best guess.
    """
    freqs = decode_bins(logits)
    if voiced is not None:
        freqs = np.where(np.asarray(voiced, dtype=bool), freqs, 0.0)
    return Contour(times=frame_times, freqs=freqs)


def pitch_head(conv_channels: int = 32, n_bins: int = N_BINS) -> FrequencyHead:
    """The pitch head -- symmetric with ``onset_head``/``voicing_head`` in ``frame``.

    Takes ``conv_channels`` rather than a backbone's ``hidden`` because it reads the
    front-end's output, not the temporal stack's: the whole point is to tap the network
    while the frequency axis is still there.
    """
    return FrequencyHead(conv_channels, n_bins)


def bin_of(midi: float) -> float:
    """The (fractional) CQT bin a MIDI pitch lands on -- the scalar form of the above."""
    return float(midi_to_bin(midi))
