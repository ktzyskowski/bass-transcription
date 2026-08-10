"""Serde: the on-disk formats -- CachedItem's split npz/.npy layout and the
contour npz -- round-trip, and the legacy migration. Atomic writes are exercised
implicitly (a save that completes leaves no .partial).
"""

import numpy as np

from stemify.music.contour import Contour
from stemify.core.legacy import migrate_example_cache
from stemify.core.serde import (
    CachedItem,
    features_npy,
    load_contour,
    save_contour,
)


def _save_example(tmp_path, name, features, target):
    ex = CachedItem(
        name, features.astype(np.float32), {"onset": target.astype(np.float32)},
        np.arange(features.shape[1]) * 0.0116,
    )
    path = tmp_path / f"{name}.npz"
    ex.save(path)
    return path


def test_example_round_trips_through_disk(tmp_path):
    path = _save_example(tmp_path, "rt", np.random.rand(5, 6), np.arange(6))
    loaded = CachedItem.load(path)
    assert loaded.name == "rt"
    assert loaded.features.shape == (5, 6)
    assert "onset" in loaded.targets


def test_load_mmap_returns_memmap_and_splits_features(tmp_path):
    feats = np.random.rand(5, 8).astype(np.float32)
    path = _save_example(tmp_path, "mm", feats, np.arange(8))
    # features live in a bare .npy sidecar, not the npz
    assert features_npy(path).exists()
    assert "features" not in np.load(path).files
    ex = CachedItem.load(path, mmap=True)
    assert isinstance(ex.features, np.memmap)          # not resident in RAM
    np.testing.assert_array_equal(np.asarray(ex.features), feats)


def test_save_leaves_no_partial(tmp_path):
    _save_example(tmp_path, "clean", np.random.rand(4, 5), np.arange(5))
    assert list(tmp_path.glob("*.partial")) == []  # atomic write cleaned up


def test_migrate_splits_legacy_npz_and_is_idempotent(tmp_path):
    legacy = tmp_path / "old.npz"
    np.savez_compressed(
        legacy, name="old", features=np.ones((4, 6), dtype=np.float32),
        frame_times=np.arange(6), target_onset=np.zeros(6),
    )
    assert migrate_example_cache(tmp_path) == 1
    assert features_npy(legacy).exists()
    assert "features" not in np.load(legacy).files    # rewritten slim
    ex = CachedItem.load(legacy)
    assert ex.features.shape == (4, 6) and "onset" in ex.targets
    assert migrate_example_cache(tmp_path) == 0        # already split -> no-op


def test_contour_round_trips_with_and_without_confidence(tmp_path):
    times = np.linspace(0, 1, 5)
    c = Contour(times=times, freqs=np.full(5, 110.0), confidence=np.full(5, 0.9))
    save_contour(c, tmp_path / "c.npz")
    back = load_contour(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.freqs, c.freqs)
    np.testing.assert_array_equal(back.confidence, c.confidence)

    bare = Contour(times=times, freqs=np.full(5, 110.0))  # no confidence
    save_contour(bare, tmp_path / "b.npz")
    assert load_contour(tmp_path / "b.npz").confidence is None
