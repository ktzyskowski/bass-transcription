"""The CQT front-end: the frame grid matches the features, extract_features pairs
them, and the FrameFeatures object wraps arrays without copying."""

import numpy as np
import soundfile as sf

from stemify.audio.features import (
    N_BINS,
    SAMPLE_RATE,
    FrameFeatures,
    extract_features,
    frame_times_for,
    stack_channels,
)


def test_frame_times_match_frame_count_and_are_evenly_spaced():
    feats = np.zeros((N_BINS, 10))
    times = frame_times_for(feats)
    assert times.shape == (10,)
    diffs = np.diff(times)
    assert np.allclose(diffs, diffs[0])          # one hop apart, uniform
    assert diffs[0] > 0


def test_features_wraps_arrays_without_copying():
    values = np.zeros((N_BINS, 5))
    times = np.arange(5, dtype=float)
    f = FrameFeatures(values, times)
    assert f.values is values                    # holds the reference, no copy
    assert f.frame_times is times


def test_extract_features_pairs_cqt_with_its_grid(tmp_path):
    wav = tmp_path / "tone.wav"
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    feats = extract_features(wav)
    assert isinstance(feats, FrameFeatures)
    assert feats.values.shape[0] == N_BINS
    # the whole point of the pairing: the grid always matches the feature length
    assert feats.frame_times.shape[0] == feats.values.shape[1]


# --------------------------------------------------------------------------- #
# stack_channels: several renderings of one track -> one model input           #
# --------------------------------------------------------------------------- #
def test_stack_channels_leaves_a_lone_channel_alone():
    # The single-channel path must not gain a stray axis or pay for a copy --
    # every existing checkpoint feeds (n_bins, T) and has to keep working.
    f = FrameFeatures(np.zeros((N_BINS, 5)), np.arange(5, dtype=float))
    assert stack_channels([f]) is f


def test_stack_channels_puts_the_channel_axis_first():
    times = np.arange(5, dtype=float)
    bass = FrameFeatures(np.full((N_BINS, 5), 1.0), times)
    mix = FrameFeatures(np.full((N_BINS, 5), 2.0), times)
    stacked = stack_channels([bass, mix])
    assert stacked.values.shape == (2, N_BINS, 5)
    assert (stacked.values[0] == 1.0).all()      # order is preserved
    assert (stacked.values[1] == 2.0).all()
    assert stacked.frame_times.shape == (5,)


def test_stack_channels_truncates_to_the_shortest():
    # Separate files for the same music can differ by a frame at the tail. Every
    # channel is a CQT on the same hop, so frame k is the same instant in all of
    # them -- the overhang is surplus, not a misalignment, and the grid must come
    # back trimmed to match or a target would be aligned against the wrong length.
    bass = FrameFeatures(np.ones((N_BINS, 7)), np.arange(7, dtype=float))
    mix = FrameFeatures(np.ones((N_BINS, 5)), np.arange(5, dtype=float))
    stacked = stack_channels([bass, mix])
    assert stacked.values.shape == (2, N_BINS, 5)
    assert stacked.frame_times.shape == (5,)
