"""Serde: the on-disk format -- CachedItem's split npz/.npy layout -- round-trips, and
the legacy migration. Atomic writes are exercised implicitly (a save that completes
leaves no .partial).
"""

import numpy as np

from stemify.persistence.serde import CachedItem, features_npy


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

