"""Dataset assembly: target building, windowing, padding, collation.

build_example is exercised on a synthetic contour so target alignment is checked
without audio; the windowing tests round-trip through a saved .npz so the cache
path is covered too.
"""


import numpy as np
import pytest
import torch

from stemify.cache import Cache
from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.serde import TrackExample
from stemify.sources import Sources
from stemify.training.datasets.frame import FrameWindowDataset, collate
from stemify.training.examples import build_example


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
    ds = FrameWindowDataset([(path,)], heads=["onset"], window_frames=4, hop_frames=2)
    assert len(ds) == 4  # starts 0, 2, 4, 6
    x, y = ds[0]
    assert x.shape == (1, 5, 4)
    assert y["onset"].shape == (4,)


def test_collate_stacks_batch(tmp_path):
    path = _save_example(tmp_path, "t", np.random.rand(5, 10), np.arange(10))
    ds = FrameWindowDataset([(path,)], heads=["onset"], window_frames=4, hop_frames=2)
    xb, yb = collate([ds[0], ds[1]])
    assert xb.shape == (2, 1, 5, 4)
    assert yb["onset"].shape == (2, 4)


def test_short_track_is_padded_not_dropped(tmp_path):
    path = _save_example(tmp_path, "s", np.ones((5, 3)), np.ones(3))
    ds = FrameWindowDataset([(path,)], heads=["onset"], window_frames=4, hop_frames=2)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (1, 5, 4)
    assert x[0, :, 3].sum() == 0.0        # feature pad is zero
    assert y["onset"][3] == 0.0           # target pad is zero


def test_bichannel_window_stacks_channels_in_order(tmp_path):
    # Constant-valued channels so a swap or a duplicated channel is visible. Only
    # channel 0 carries targets, which is what the training loader guarantees.
    bass = _save_example(tmp_path, "bass", np.full((5, 10), 1.0), np.arange(10))
    mix = _save_example(tmp_path, "mix", np.full((5, 10), 2.0), np.zeros(10))
    ds = FrameWindowDataset(
        [(bass, mix)], heads=["onset"], window_frames=4, hop_frames=2
    )
    x, y = ds[0]
    assert x.shape == (2, 5, 4)
    assert (x[0] == 1.0).all()          # channel 0 is the bass
    assert (x[1] == 2.0).all()          # channel 1 is the mixture
    # Targets come from channel 0 alone -- the mixture's npz is read for features
    # only, so its (here all-zero) targets must not reach the batch.
    assert (y["onset"] == torch.arange(4, dtype=torch.float32)).all()


def test_bichannel_collate_keeps_the_channel_axis(tmp_path):
    bass = _save_example(tmp_path, "bass", np.full((5, 10), 1.0), np.arange(10))
    mix = _save_example(tmp_path, "mix", np.full((5, 10), 2.0), np.arange(10))
    ds = FrameWindowDataset(
        [(bass, mix)], heads=["onset"], window_frames=4, hop_frames=2
    )
    xb, _ = collate([ds[0], ds[1]])
    assert xb.shape == (2, 2, 5, 4)     # (B, C, n_bins, W)


def test_channels_of_different_lengths_are_rejected(tmp_path):
    # Renderings of one track share a CQT hop, so their frame counts match exactly.
    # A mismatch means the channels were paired wrongly -- and that failure is
    # otherwise invisible: labels still fit channel 0 and the loss still falls.
    bass = _save_example(tmp_path, "bass", np.ones((5, 10)), np.arange(10))
    other = _save_example(tmp_path, "other", np.ones((5, 7)), np.arange(7))
    with pytest.raises(ValueError, match="disagree on length"):
        FrameWindowDataset([(bass, other)], heads=["onset"], window_frames=4)


def test_prepare_backfills_missing_target_without_recomputing_features(
    tmp_path, monkeypatch
):
    from stemify.contour import Contour
    from stemify.serde import features_npy
    from stemify.training.examples import prepare_examples
    from stemify.transforms import Identity

    # Pre-save an onset-only example, as a prior (onset-only) run would have. The
    # cache decides the filename, so the backfill has to find it by the same key the
    # writer used -- the resolved audio path.
    audio = tmp_path / "trk.wav"
    audio.write_text("x")
    cache = Cache(tmp_path / "cache")
    ft = np.arange(8) * 0.0116
    TrackExample(
        "trk", np.random.rand(5, 8).astype(np.float32),
        {"onset": np.zeros(8, np.float32)}, ft,
    ).save(cache.feature_path(audio))
    path = cache.feature_path(audio)
    feat_mtime = features_npy(path).stat().st_mtime_ns  # CQT sidecar must not change

    ref = Contour(ft, np.where(np.arange(8) < 4, 110.0, 0.0))  # voiced first half

    # The point of the test: the backfill must not reach the CQT. (audio() itself is
    # now called for the cache *key*, but that is a path lookup, not a decode.)
    def _no_cqt(path):
        raise AssertionError("backfill must not recompute the CQT")

    monkeypatch.setattr("stemify.cache.extract_features", _no_cqt)

    class _Item:
        name = "trk"
        def audio(self):
            return audio
        def sources(self):
            return Sources(bass=audio)
        def reference(self):
            return ref
        def reference_notes(self):
            return [NoteEvent(onset=0.0, offset=0.05, midi_pitch=45.0)]

    # Requesting a head the cache lacks backfills it from the reference alone.
    prepare_examples([_Item()], ["onset", "voicing"], cache, Identity())

    loaded = TrackExample.load(path)
    assert "voicing" in loaded.targets                          # added
    assert "onset" in loaded.targets                            # kept
    assert loaded.targets["voicing"][:4].all()                 # voiced first half
    assert features_npy(path).stat().st_mtime_ns == feat_mtime  # features untouched


def test_two_items_sharing_audio_but_not_targets_is_an_error(tmp_path, monkeypatch):
    # Content-addressing means one audio file -> one cache entry. That is correct for
    # the CQT but not for targets, so two *different* items pointing at the same audio
    # must fail loudly rather than let the second item's labels vanish. Slakh cannot
    # produce this today (mix sources are filtered to single-bass tracks); the guard
    # exists so relaxing that filter cannot silently mislabel training data.
    import pytest

    from stemify.cache import Cache
    from stemify.contour import Contour
    from stemify.training.examples import prepare_examples
    from stemify.transforms import Identity

    audio = tmp_path / "mix.wav"
    audio.write_text("x")
    ft = np.arange(8) * 0.0116

    def fake_extract(path):
        from stemify.features import FrameFeatures
        return FrameFeatures(np.random.rand(5, 8).astype(np.float32), ft)

    monkeypatch.setattr("stemify.cache.extract_features", fake_extract)

    class _Item:
        def __init__(self, name):
            self.name = name
        def audio(self):
            return audio            # both items resolve to the SAME file
        def sources(self):
            return Sources(bass=audio)
        def reference(self):
            return Contour(ft, np.full(8, 110.0))
        def reference_notes(self):
            return [NoteEvent(onset=0.0, offset=0.05, midi_pitch=45.0)]

    with pytest.raises(ValueError, match="share cached features"):
        prepare_examples(
            [_Item("Track01/S00"), _Item("Track01/S01")],
            ["onset"], Cache(tmp_path / "cache"), Identity(),
        )
