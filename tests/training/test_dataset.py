"""Dataset assembly: target building, windowing, padding, collation.

build_example is exercised on a synthetic contour so target alignment is checked
without audio; the windowing tests round-trip through a saved .npz so the cache
path is covered too.
"""

import numpy as np

from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.training.dataset import (
    FrameWindowDataset,
    TrackExample,
    build_example,
    collate,
)


def test_build_example_aligns_onset_and_voicing():
    frame_times = np.arange(20) * 0.0116
    ref_times = np.arange(24) * 0.01
    freqs = np.where(np.arange(24) < 12, 110.0, 0.0)  # voiced first ~0.12 s
    reference = Contour(ref_times, freqs)
    notes = [NoteEvent(onset=0.0, offset=0.12, midi_pitch=45.0)]

    ex = build_example(
        "t", np.zeros((5, 20), dtype=np.float32), frame_times,
        reference, notes, heads=["onset", "voicing"],
    )
    assert set(ex.targets) == {"onset", "voicing"}
    assert ex.targets["onset"].shape == (20,)
    assert ex.targets["onset"][:3].any()      # onset marked near t=0
    assert ex.targets["voicing"][:5].all()    # voiced early
    assert not ex.targets["voicing"][-3:].any()  # rest late


def _save_example(tmp_path, name, features, target):
    ex = TrackExample(
        name, features.astype(np.float32), {"onset": target.astype(np.float32)},
        np.arange(features.shape[1]) * 0.0116,
    )
    path = tmp_path / f"{name}.npz"
    ex.save(path)
    return path


def test_windowing_enumerates_strided_windows(tmp_path):
    path = _save_example(tmp_path, "t", np.random.rand(5, 10), np.arange(10))
    ds = FrameWindowDataset([path], heads=["onset"], window_frames=4, hop_frames=2)
    assert len(ds) == 4  # starts 0, 2, 4, 6
    x, y = ds[0]
    assert x.shape == (1, 5, 4)
    assert y["onset"].shape == (4,)


def test_collate_stacks_batch(tmp_path):
    path = _save_example(tmp_path, "t", np.random.rand(5, 10), np.arange(10))
    ds = FrameWindowDataset([path], heads=["onset"], window_frames=4, hop_frames=2)
    xb, yb = collate([ds[0], ds[1]])
    assert xb.shape == (2, 1, 5, 4)
    assert yb["onset"].shape == (2, 4)


def test_short_track_is_padded_not_dropped(tmp_path):
    path = _save_example(tmp_path, "s", np.ones((5, 3)), np.ones(3))
    ds = FrameWindowDataset([path], heads=["onset"], window_frames=4, hop_frames=2)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (1, 5, 4)
    assert x[0, :, 3].sum() == 0.0        # feature pad is zero
    assert y["onset"][3] == 0.0           # target pad is zero


def test_example_round_trips_through_disk(tmp_path):
    path = _save_example(tmp_path, "rt", np.random.rand(5, 6), np.arange(6))
    loaded = TrackExample.load(path)
    assert loaded.name == "rt"
    assert loaded.features.shape == (5, 6)
    assert "onset" in loaded.targets