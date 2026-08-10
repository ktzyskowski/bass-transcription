"""Dataset assembly: target building, windowing, padding, collation.

build_example is exercised on a synthetic contour so target alignment is checked
without audio; the windowing tests round-trip through a saved .npz so the cache
path is covered too.
"""


import numpy as np
import pytest
import torch

from stemify.audio.features import SILENCE_DB
from stemify.audio.cache import feature_path
from stemify.persistence.store import Store
from stemify.music.contour import Contour
from stemify.music.notes import NoteEvent
from stemify.persistence.serde import CachedItem
from stemify.audio.sources import Sources
from stemify.core.examples import build_example
from stemify.training.frame.dataset import FrameWindowDataset, collate
from stemify.training.frame.recipe import FrameRecipe


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
    ex = CachedItem(
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
    batch = collate([ds[0], ds[1]])
    xb, yb = batch.x, batch.y
    assert xb.shape == (2, 1, 5, 4)
    assert yb["onset"].shape == (2, 4)


def test_short_track_is_padded_not_dropped(tmp_path):
    path = _save_example(tmp_path, "s", np.ones((5, 3)), np.ones(3))
    ds = FrameWindowDataset([(path,)], heads=["onset"], window_frames=4, hop_frames=2)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (1, 5, 4)
    # Feature pad is SILENCE_DB, not zero: the CQT is dB relative to the track's own
    # peak, so a zero-filled tail would be the loudest thing in the window.
    assert (x[0, :, 3] == SILENCE_DB).all()
    assert y["onset"][3] == 0.0           # target pad is zero -- "no onset" is a label


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


def _bichannel(tmp_path, **kwargs) -> FrameWindowDataset:
    bass = _save_example(tmp_path, "bass", np.full((5, 10), 1.0), np.arange(10))
    mix = _save_example(tmp_path, "mix", np.full((5, 10), 2.0), np.zeros(10))
    return FrameWindowDataset(
        [(bass, mix)], heads=["onset"], window_frames=4, hop_frames=2, **kwargs
    )


def test_stem_dropout_silences_the_context_channel_and_never_the_bass(tmp_path):
    # p=1 so the draw is not being tested, only what a drop *does*: the context
    # channel becomes SILENCE_DB (the dB floor) rather than zero, which in this
    # feature space would be full-scale energy -- the opposite of an absent input.
    x, y = _bichannel(tmp_path, stem_dropout=1.0)[0]
    assert (x[0] == 1.0).all(), "channel 0 is the bass and must never be dropped"
    assert (x[1] == SILENCE_DB).all()
    # Dropping a channel must not touch the labels: the question is unchanged, only
    # the evidence available to answer it.
    assert (y["onset"] == torch.arange(4, dtype=torch.float32)).all()


def test_stem_dropout_defaults_to_off(tmp_path):
    # Load-bearing: it changes the input distribution, so every arm trained before it
    # existed must rebuild identically. A default that silently augmented would make
    # those runs incomparable with no error anywhere.
    x, _ = _bichannel(tmp_path)[0]
    assert (x[0] == 1.0).all()
    assert (x[1] == 2.0).all()


def test_stem_dropout_rejects_a_probability_outside_the_unit_interval(tmp_path):
    # A fraction typed as a percentage is the plausible slip, and at p=50 every
    # context channel is silenced on every window -- which trains a model that
    # ignores the mixture while reporting the flag as set.
    with pytest.raises(ValueError, match=r"stem_dropout must be in \[0, 1\]"):
        _bichannel(tmp_path, stem_dropout=50.0)


def test_bichannel_collate_keeps_the_channel_axis(tmp_path):
    bass = _save_example(tmp_path, "bass", np.full((5, 10), 1.0), np.arange(10))
    mix = _save_example(tmp_path, "mix", np.full((5, 10), 2.0), np.arange(10))
    ds = FrameWindowDataset(
        [(bass, mix)], heads=["onset"], window_frames=4, hop_frames=2
    )
    xb = collate([ds[0], ds[1]]).x
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
    from stemify.music.contour import Contour
    from stemify.persistence.serde import features_npy
    from stemify.core.examples import prepare_examples
    from stemify.audio.transforms import Identity

    # Pre-save an onset-only example, as a prior (onset-only) run would have. The
    # cache decides the filename, so the backfill has to find it by the same key the
    # writer used -- the resolved audio path.
    audio = tmp_path / "trk.wav"
    audio.write_text("x")
    cache = Store(tmp_path / "cache")
    ft = np.arange(8) * 0.0116
    CachedItem(
        "trk", np.random.rand(5, 8).astype(np.float32),
        {"onset": np.zeros(8, np.float32)}, ft,
    ).save(feature_path(cache, audio))
    path = feature_path(cache, audio)
    feat_mtime = features_npy(path).stat().st_mtime_ns  # CQT sidecar must not change

    ref = Contour(ft, np.where(np.arange(8) < 4, 110.0, 0.0))  # voiced first half

    # The point of the test: the backfill must not reach the CQT. (audio() itself is
    # now called for the cache *key*, but that is a path lookup, not a decode.)
    def _no_cqt(path):
        raise AssertionError("backfill must not recompute the CQT")

    monkeypatch.setattr("stemify.audio.cache.extract_features", _no_cqt)

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

    loaded = CachedItem.load(path)
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

    from stemify.persistence.store import Store
    from stemify.music.contour import Contour
    from stemify.core.examples import prepare_examples
    from stemify.audio.transforms import Identity

    audio = tmp_path / "mix.wav"
    audio.write_text("x")
    ft = np.arange(8) * 0.0116

    def fake_extract(path):
        from stemify.audio.features import FrameFeatures
        return FrameFeatures(np.random.rand(5, 8).astype(np.float32), ft)

    monkeypatch.setattr("stemify.audio.cache.extract_features", fake_extract)

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
            ["onset"], Store(tmp_path / "cache"), Identity(),
        )


def test_parse_pos_weight_rejects_unknown_heads():
    # A typo must not pass silently. FrameBCELoss reads weights with .get(head), so an
    # unrecognised key is never looked up: the run trains unweighted while its logged
    # params claim otherwise -- which is exactly how pos_weight sat declared, plumbed
    # through the loss, and never set by anything.
    import pytest

    from stemify.training.cli import _parse_pos_weight

    assert _parse_pos_weight("") == {}
    assert _parse_pos_weight("onset=5") == {"onset": 5.0}
    assert _parse_pos_weight("onset=5,voicing=2") == {"onset": 5.0, "voicing": 2.0}
    for bad in ("onsett=5", "onset", "pitch=3"):
        with pytest.raises(ValueError, match="--pos-weight"):
            _parse_pos_weight(bad)


def test_pos_weight_actually_reaches_the_loss():
    # The end-to-end version of the same concern: a weight that is parsed but never
    # applied would leave the "optimistic encoder" arm identical to the balanced one.
    import torch

    from stemify.training.frame.dataset import FrameBatch
    from stemify.training.frame.loss import FrameBCELoss

    x = torch.zeros(2, 1, 4, 8)
    y = {"onset": torch.zeros(2, 8)}
    y["onset"][:, 0] = 1.0

    class _Const(torch.nn.Module):
        def forward(self, _x):
            return {"onset": torch.zeros(2, 8, 1)}

    plain = FrameBCELoss(["onset"], {})
    heavy = FrameBCELoss(["onset"], {"onset": 5.0})
    _, m_plain = plain.compute(_Const(), FrameBatch(x, y), torch.device("cpu"))
    _, m_heavy = heavy.compute(_Const(), FrameBatch(x, y), torch.device("cpu"))
    assert m_heavy["total"] > m_plain["total"], (
        "a positive weight must increase the loss on a missed positive"
    )


def test_val_conditions_accept_retired_names():
    # The exact crash that killed an overnight arm: --val-sources with old tags.
    import inspect

    from stemify.training import train as train_mod

    src = inspect.getsource(train_mod.train)
    assert "canonical(t) for t in val_conditions" in src, (
        "val conditions must go through the legacy mapping, like --sources does"
    )
    # And the defaults must name conditions that actually exist.
    from stemify.core.conditions import CONDITIONS

    for default in ("bichannel-clean", "bichannel-separated", "clean", "mixture",
                    "separated"):
        assert default in CONDITIONS


def test_every_recipe_exposes_loggable_params():
    # The run's params dict is assembled from ``recipe.params`` because the entry point
    # cannot know which knobs a strategy has. Reaching for one directly is how
    # ``config.window_frames`` survived the move onto FrameRecipe and crashed the next
    # launch -- after the tests passed. Every registered recipe must answer.
    from stemify.training.recipes import RECIPES

    for name, cls in RECIPES.items():
        params = cls().params
        assert isinstance(params, dict), name
        assert all(isinstance(k, str) for k in params), name


def test_frame_params_report_the_knobs_the_recipe_owns():
    # These moved off TrainConfig; a run that does not record them cannot be told
    # apart from one trained with different windows, weights or input features.
    r = FrameRecipe(pitch=True, window_frames=64, hop_frames=32,
                    pos_weight={"onset": 5.0}, harmonics=[1, 2, 3, 4])
    assert r.params == {
        "window": 64, "hop": 32, "pitch": True, "pos_weight": "onset=5.0",
        "harmonics": "1,2,3,4",
    }
    assert FrameRecipe().params["pos_weight"] == "none"
    # The plain CQT is still *recorded*, not omitted: "1" and a missing key would
    # otherwise be indistinguishable when comparing an old run against a new one.
    assert FrameRecipe().params["harmonics"] == "1"


def test_open_track_cache_is_bounded(tmp_path):
    # Features are memmapped, but each CachedItem's targets and frame_times are
    # resident and every DataLoader worker is a fork with its own copy. Unbounded,
    # a worker converges on the whole corpus: a 2-channel pitch run OOM'd a 15 GB
    # host after one epoch. Nothing here checks speed -- only that it stops growing.
    from stemify.training.frame.dataset import CACHE_TRACKS

    paths = [
        (_save_example(tmp_path, f"t{i}", np.full((4, 8), float(i)), np.arange(8)),)
        for i in range(CACHE_TRACKS + 5)
    ]
    ds = FrameWindowDataset(paths, heads=["onset"], window_frames=4, hop_frames=4)
    for i in range(len(ds)):
        ds[i]
    assert len(ds._cache) <= CACHE_TRACKS

    # Eviction must not corrupt what it serves: track 0 is long gone from the cache
    # by now, so this re-reads it from disk and must still give back track 0's data.
    x, y = ds[0]
    assert x.shape == (1, 4, 4)
    assert (x[0] == 0.0).all()
    assert (y["onset"] == torch.arange(4, dtype=torch.float32)).all()
