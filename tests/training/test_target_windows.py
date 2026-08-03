"""Per-frame targets sliced to a chunk window, and why they pad with NaN.

A chunk window runs off both ends of a track -- before frame 0 for its left context,
past the last frame for its right margin -- and those positions have no label. The
padding value is therefore load-bearing in a way the *feature* padding is not: zero is
a meaningless CQT magnitude but a perfectly meaningful MIDI pitch, so zero-padding
would teach the pitch head to predict MIDI 0 at every track edge while the loss went
down. NaN is what ``pitch_ce`` masks on, so a padded frame contributes nothing.

The gradient test at the bottom is the one that matters: it checks the *consequence*
rather than the representation, so it still holds if the masking moves.
"""

import numpy as np
import pytest
import torch

from stemify.serde import TrackExample
from stemify.models.heads import pitch_ce
from stemify.training.semicrf.tracks import target_windows

N_BINS, HOP_S, T = 12, 0.0116, 20


def _example(midi: float = 45.0) -> TrackExample:
    return TrackExample(
        "t0",
        np.zeros((N_BINS, T), dtype=np.float32),
        {"pitch": np.full(T, midi, dtype=np.float32)},
        np.arange(T) * HOP_S,
    )


def test_a_window_inside_the_track_is_all_real_labels():
    y = target_windows(_example(), ["pitch"], start=5, width=10)["pitch"]
    assert y.shape == (10,)
    assert not np.isnan(y).any()
    assert (y == 45.0).all()


def test_running_off_either_end_pads_with_nan_not_zero():
    # The whole point. Zero is a valid MIDI value, so a zero here is indistinguishable
    # from a real label and the loss would train toward it.
    y = target_windows(_example(), ["pitch"], start=-3, width=T + 6)["pitch"]
    assert np.isnan(y[:3]).all(), "left pad must be NaN"
    assert np.isnan(y[-3:]).all(), "right pad must be NaN"
    assert (y[3:-3] == 45.0).all()
    assert not (y[:3] == 0.0).any(), "zero-padded: MIDI 0 is a label, not an absence"


def test_a_window_entirely_off_the_track_is_all_nan():
    y = target_windows(_example(), ["pitch"], start=-50, width=10)["pitch"]
    assert np.isnan(y).all()


def test_a_missing_head_names_itself():
    # The dataset must be built with the heads it will serve; failing here beats
    # serving a silently absent target.
    with pytest.raises(KeyError, match="pitch"):
        target_windows(TrackExample("t", np.zeros((N_BINS, T)), {}, np.arange(T)),
                       ["pitch"], start=0, width=4)


def test_padded_frames_contribute_no_gradient():
    # The consequence, not the representation: whatever the padding value is, the
    # frames off the end of the track must not move the pitch head. If padding ever
    # regressed to zeros this fails, because MIDI 0 would pull bin 0 upward.
    width, n_bins = 12, 216
    logits = torch.zeros(1, width, n_bins, requires_grad=True)

    y = target_windows(_example(), ["pitch"], start=-4, width=width)["pitch"]
    assert np.isnan(y[:4]).all() and not np.isnan(y[4:]).any()

    pitch_ce(logits, torch.from_numpy(y)[None]).backward()
    grad = logits.grad
    assert grad is not None
    assert torch.count_nonzero(grad[0, :4]) == 0, "a padded frame moved the head"
    assert torch.count_nonzero(grad[0, 4:]) > 0, "the real frames trained nothing"
