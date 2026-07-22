"""The CQT front-end: the frame grid matches the features, extract_features pairs
them, and the Features object wraps arrays without copying."""

import numpy as np
import soundfile as sf

from stemify.features import (
    N_BINS,
    SAMPLE_RATE,
    Features,
    extract_features,
    frame_times_for,
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
    f = Features(values, times)
    assert f.values is values                    # holds the reference, no copy
    assert f.frame_times is times


def test_extract_features_pairs_cqt_with_its_grid(tmp_path):
    wav = tmp_path / "tone.wav"
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    feats = extract_features(wav)
    assert isinstance(feats, Features)
    assert feats.values.shape[0] == N_BINS
    # the whole point of the pairing: the grid always matches the feature length
    assert feats.frame_times.shape[0] == feats.values.shape[1]
