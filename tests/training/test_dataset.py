"""Dataset assembly: target building, windowing, padding, collation.

build_example is exercised on a synthetic contour so target alignment is checked
without audio; the windowing tests round-trip through a saved .npz so the cache
path is covered too.
"""

import numpy as np

from stemify.contour import Contour
from stemify.notes import NoteEvent
from pathlib import Path

from stemify.training.dataset import (
    FrameWindowDataset,
    TrackExample,
    build_example,
    collate,
    example_path,
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


def test_load_mmap_returns_memmap_and_splits_features(tmp_path):
    from stemify.training.dataset import features_npy

    feats = np.random.rand(5, 8).astype(np.float32)
    path = _save_example(tmp_path, "mm", feats, np.arange(8))
    # features live in a bare .npy sidecar, not the npz
    assert features_npy(path).exists()
    assert "features" not in np.load(path).files
    ex = TrackExample.load(path, mmap=True)
    assert isinstance(ex.features, np.memmap)          # not resident in RAM
    np.testing.assert_array_equal(np.asarray(ex.features), feats)


def test_migrate_splits_legacy_npz_and_is_idempotent(tmp_path):
    from stemify.training.dataset import features_npy, migrate_example_cache

    legacy = tmp_path / "old.npz"
    np.savez_compressed(
        legacy, name="old", features=np.ones((4, 6), dtype=np.float32),
        frame_times=np.arange(6), target_onset=np.zeros(6),
    )
    assert migrate_example_cache(tmp_path) == 1
    assert features_npy(legacy).exists()
    assert "features" not in np.load(legacy).files    # rewritten slim
    ex = TrackExample.load(legacy)
    assert ex.features.shape == (4, 6) and "onset" in ex.targets
    assert migrate_example_cache(tmp_path) == 0        # already split -> no-op


def test_example_path_keys_on_name_and_transform():
    # The writer (prepare_examples) and reader (OnsetValidator) must agree on this
    # filename, so it lives in one helper. '/' in a track name is flattened, and
    # the transform is part of the key so stem and demucs of one track don't collide.
    cache = Path("cache")
    assert example_path(cache, "Track01", "identity").name == "Track01__identity.npz"
    assert example_path(cache, "a/b", "demucs").name == "a_b__demucs.npz"
    stem = example_path(cache, "Track01", "identity")
    demucs = example_path(cache, "Track01", "demucs-htdemucs-bass")
    assert stem != demucs  # same track, different transform -> distinct files