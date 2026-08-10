"""Harmonic stacking: the same CQT, re-indexed so one kernel can see a note's partials.

A string at ``f`` also radiates ``2f``, ``3f``, ``4f``. On a log-frequency axis those
sit a *constant* number of bins above the fundamental -- +36, +57, +72 at 36 bins per
octave -- so one note's evidence is scattered across the frequency axis at fixed
offsets. The front-end is two 3x3 convolutions, a +-2 bin reach, and then it flattens
frequency away entirely: as laid out, a fundamental and its own harmonics can never
appear in one kernel's window.

Stacking slices them into channels at a shared ``(bin, frame)`` position. No information
is added -- these are literally the same numbers -- only adjacency, which is the whole
of what a convolution can exploit. Measured on the val set, the fundamental's flux peak
lands within one frame of a true onset 9.6% of the time against 48% at ``h=4``.
"""

import numpy as np
import torch

from stemify.audio.features import BINS_PER_OCTAVE, FMIN, N_BINS, harmonic_shifts
from stemify.models.heads.pitch import bin_of, pitch_ce
from stemify.models.networks import TCN, _ConvFrontend


# --------------------------------------------------------------------------- #
# The geometry                                                                #
# --------------------------------------------------------------------------- #
def test_shifts_land_on_the_harmonic_frequencies():
    """The claim the whole design rests on: ``h * f`` is a fixed bin offset from ``f``,
    the same offset at every pitch. Checked against the frequencies themselves rather
    than against a hardcoded table, which would only restate the implementation."""
    for h, shift in zip((1, 2, 3, 4, 5), harmonic_shifts((1, 2, 3, 4, 5)), strict=True):
        for midi_bin in (0, 40, 84):        # bottom, mid-bass, top of the bass range
            f = FMIN * 2 ** (midi_bin / BINS_PER_OCTAVE)
            want = BINS_PER_OCTAVE * np.log2(h * f / FMIN)
            assert abs((midi_bin + shift) - want) <= 0.5, (
                f"h={h} at bin {midi_bin}: shift {shift} lands {midi_bin + shift}, "
                f"the harmonic is at {want:.2f}"
            )


def test_the_octave_is_exactly_one_octave_of_bins():
    assert harmonic_shifts((1, 2, 4))[1] == BINS_PER_OCTAVE
    assert harmonic_shifts((1, 2, 4))[2] == 2 * BINS_PER_OCTAVE


# --------------------------------------------------------------------------- #
# The stack                                                                   #
# --------------------------------------------------------------------------- #
def test_stacking_puts_a_notes_partials_at_one_position():
    """The property, stated as the thing it is for: a synthetic note with energy at its
    fundamental *and* its harmonics becomes a column that is hot in every channel at a
    single bin -- reachable by one 3x3 kernel."""
    fe = _ConvFrontend(harmonics=(1, 2, 3, 4))
    b0, T = 40, 5
    x = torch.zeros(1, 1, N_BINS, T)
    # A *distinct* amplitude per partial, which is what makes this test bite. With
    # every harmonic set to 1.0 the assertion below passes even for a stack that
    # ignores its shifts entirely and copies one slice into all four channels -- the
    # fundamental alone would make the column look right.
    for k, shift in enumerate(harmonic_shifts((1, 2, 3, 4))):
        x[0, 0, b0 + shift] = float(k + 1)   # the note, as physics leaves it: scattered

    stacked = fe.stack(x)
    assert stacked.shape == (1, 4, fe.n_bins, T)
    # ...and as the network now sees it: one bin, carrying every partial in order.
    assert torch.equal(stacked[0, :, b0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # Nothing else in the column lit up, or the "position" would not be informative.
    assert float(stacked[0, :, b0 + 1].sum()) == 0.0


def test_the_default_is_the_plain_cqt_untouched():
    """`harmonics=(1,)` must be the *identity*, not merely equivalent. Five checkpoints
    were trained before this existed and rebuild through this code path; a stack that
    copied or reordered even losslessly would still change what their weights index."""
    fe = _ConvFrontend()
    x = torch.randn(2, 1, N_BINS, 7)
    assert fe.stack(x) is x
    assert fe.n_bins == N_BINS
    assert fe.conv[0].in_channels == 1


def test_every_stacked_channel_is_real_data():
    """The fundamental axis is cropped by the topmost shift so no channel needs padding.
    A stack that kept the full height would feed the conv a block of zeros above the
    crop line and teach it that harmonics vanish there."""
    for harmonics in [(1, 2), (1, 2, 3, 4), (1, 2, 3, 4, 5)]:
        fe = _ConvFrontend(harmonics=harmonics)
        x = torch.randn(1, 1, N_BINS, 3)
        stacked = fe.stack(x)
        assert fe.n_bins == N_BINS - max(harmonic_shifts(harmonics))
        for k, shift in enumerate(harmonic_shifts(harmonics)):
            assert torch.equal(stacked[0, k], x[0, 0, shift: shift + fe.n_bins])


def test_two_renderings_stack_harmonic_major():
    """With a mix channel the width is `renderings * harmonics`, grouped by harmonic.
    Pinned because the conv's first layer is the only thing that knows the order, and a
    silent transposition there pairs the bass's fundamental with the mix's octave."""
    fe = _ConvFrontend(in_channels=2, harmonics=(1, 2))
    x = torch.randn(1, 2, N_BINS, 4)
    stacked = fe.stack(x)
    assert stacked.shape == (1, 4, fe.n_bins, 4)
    assert torch.equal(stacked[0, 0], x[0, 0, : fe.n_bins])          # bass,  h=1
    assert torch.equal(stacked[0, 1], x[0, 1, : fe.n_bins])          # mix,   h=1
    assert torch.equal(stacked[0, 2], x[0, 0, BINS_PER_OCTAVE:])     # bass,  h=2
    assert torch.equal(stacked[0, 3], x[0, 1, BINS_PER_OCTAVE:])     # mix,   h=2


# --------------------------------------------------------------------------- #
# What the rest of the network sees                                           #
# --------------------------------------------------------------------------- #
def test_the_backbone_runs_and_keeps_its_frame_grid():
    """Stacking must not touch the time axis: every head and the whole decoder index
    frames, so a frontend that quietly changed T would misalign every target."""
    net = TCN(in_channels=1, harmonics=(1, 2, 3, 4))
    x = torch.randn(2, 1, N_BINS, 16)
    frequency, frame = net.frequency_and_frame(x)
    assert frame.shape == (2, 16, net.hidden)
    assert frequency.shape[-1] == 16
    assert frequency.shape[2] == net.frontend.n_bins


def test_the_stack_is_cheaper_than_the_axis_it_replaces():
    """Not a micro-benchmark -- a shape claim with a cost consequence. Spending the top
    of the fundamental axis on channels *shrinks* the flatten, which is what feeds the
    1x1 projection, so the harmonic model is smaller than the plain one rather than
    being a cost to justify."""
    plain = TCN(in_channels=1)
    stacked = TCN(in_channels=1, harmonics=(1, 2, 3, 4))
    assert stacked.frontend.out_channels < plain.frontend.out_channels
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(stacked) < n(plain)


def test_the_pitch_loss_follows_the_head_rather_than_the_cqt():
    """`pitch_ce` built its two-hot target at `N_BINS` regardless of the head's width,
    which a cropped head makes wrong -- and unequal widths do not broadcast, so a
    harmonic-stacked run would not have trained at all.

    Asserted through the loss's own minimum rather than just "it is finite": the target
    has to land on the *right* bin of the cropped axis, not merely on an axis of the
    right length."""
    fe = _ConvFrontend(harmonics=(1, 2, 3, 4))
    midi = torch.full((2, 5), 45.0)
    b = int(round(bin_of(45.0)))
    assert b < fe.n_bins, "the test pitch must survive the crop or it proves nothing"

    peaked = torch.full((2, 5, fe.n_bins), -10.0)
    peaked[..., b] = 10.0
    elsewhere = torch.full((2, 5, fe.n_bins), -10.0)
    elsewhere[..., b + 20] = 10.0
    assert float(pitch_ce(peaked, midi)) < float(pitch_ce(elsewhere, midi))
    assert float(pitch_ce(peaked, midi)) < 1.0
