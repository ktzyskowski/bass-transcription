"""Conditions: which audio each model input channel reads, and how they pair up.

The properties worth pinning are the ones whose failure is *silent*. A condition
that pairs track i's bass with track j's mixture still trains -- labels fit channel
0, loss still falls -- so ``channel_paths`` transposing correctly is asserted
directly. Same for the track filter: feeding a two-bass mixture to a model scored
against one bass produces a plausible number for an impossible task.
"""

from pathlib import Path

import numpy as np
import pytest

from stemify.cache import Cache
from stemify.notes import NoteEvent
from stemify.sources import Sources
from stemify.training.conditions import CONDITIONS, Channel, Condition, resolve
from stemify.transforms import AudioTransform, Identity


class _FakeSeparator(AudioTransform):
    """Resolves to a file distinct from its source, the way Demucs does."""

    @property
    def name(self) -> str:
        return "fake-sep"

    def _run(self, source, dest) -> None:
        dest.write_text("separated")


class _Item:
    """A track offering a bass and the mixture it sits inside."""

    def __init__(self, name, bass, mix):
        self.name = name
        self._bass, self._mix = bass, mix

    def audio(self):
        return self._bass

    def sources(self):
        return Sources(bass=self._bass, mix=self._mix)

    def reference_notes(self):
        return [NoteEvent(onset=0.0, offset=0.05, midi_pitch=45.0)]


# --------------------------------------------------------------------------- #
# the table                                                                    #
# --------------------------------------------------------------------------- #
def test_single_channel_conditions_stay_single_channel():
    # The three pre-existing tags must not have gained a channel: every earlier
    # result is quoted by tag, so their meaning has to be unchanged.
    for tag in ("stem", "mix", "demucs"):
        assert CONDITIONS[tag].in_channels == 1


def test_bichannel_conditions_put_the_bass_first():
    # Channel 0 is what the contour and every target describe. Reversed, the model
    # would be trained to transcribe the mixture with the bass as context.
    for tag in ("stem-mix", "demucs-mix"):
        condition = CONDITIONS[tag]
        assert condition.in_channels == 2
        assert condition.primary is condition.channels[0]
        assert condition.primary.role == "bass"
        assert condition.channels[1].role == "mix"


@pytest.mark.parametrize(
    "tag, expected",
    [("stem", False), ("mix", True), ("demucs", True),
     ("stem-mix", True), ("demucs-mix", True)],
)
def test_mix_bearing_conditions_keep_only_single_bass_tracks(tag, expected):
    # With two basses sounding, one reference contour cannot describe the mixture.
    # "stem-mix" is the case the old source-only rule would have missed: its Slakh
    # source is "stem", so nothing about the tag looks mix-flavoured.
    assert CONDITIONS[tag].unambiguous_only is expected


# --------------------------------------------------------------------------- #
# resolve                                                                      #
# --------------------------------------------------------------------------- #
def test_resolve_rejects_mixed_channel_widths():
    # One model has one input shape. Concatenated, these would yield batches whose
    # channel count depends on which track was drawn.
    with pytest.raises(ValueError, match="agree on input channel count"):
        resolve(["stem", "demucs-mix"])


def test_resolve_accepts_a_uniform_width_set():
    assert [c.name for c in resolve(["stem-mix", "demucs-mix"])] == [
        "stem-mix", "demucs-mix"
    ]


def test_resolve_rejects_an_unknown_tag():
    with pytest.raises(ValueError, match="unknown condition"):
        resolve(["stem", "bass+drums"])


# --------------------------------------------------------------------------- #
# channel resolution                                                           #
# --------------------------------------------------------------------------- #
def test_channels_reading_one_file_through_different_transforms_stay_distinct(tmp_path):
    # "demucs-mix" is exactly this: both channels read the track's mix.flac and
    # differ only in the transform. If they collided, the model would be handed the
    # same array twice and the second channel would carry no information at all.
    mix = tmp_path / "mix.wav"
    mix.write_text("x")
    item = _Item("trk", mix, mix)
    separated = Channel("bass", _FakeSeparator(tmp_path / "t")).audio(item)
    raw = Channel("mix", Identity()).audio(item)
    assert separated != raw
    assert raw == mix


def test_a_channel_whose_role_the_track_lacks_fails_loudly(tmp_path):
    bass = tmp_path / "stem.wav"
    bass.write_text("x")
    item = _Item("trk", bass, None)
    with pytest.raises(ValueError, match="no 'mix' audio"):
        Channel("mix", Identity()).audio(item)


def test_channel_paths_pairs_each_track_with_its_own_channels(tmp_path, monkeypatch):
    # The failure this guards is silent: mispaired channels leave the labels fitting
    # channel 0, so the loss still falls and the mixture channel is simply noise
    # from another song. Distinct per-file features let us assert the pairing.
    from stemify import features as features_mod

    def fake_extract(path):
        # Encode the filename in the features so a swap would be detectable.
        seed = sum(Path(path).name.encode())
        return features_mod.FrameFeatures(
            np.full((4, 6), float(seed), dtype=np.float32), np.arange(6) * 0.0116
        )

    monkeypatch.setattr("stemify.cache.extract_features", fake_extract)

    items = []
    for i in range(3):
        bass, mix = tmp_path / f"b{i}.wav", tmp_path / f"m{i}.wav"
        bass.write_text("b")
        mix.write_text("m")
        items.append(_Item(f"trk{i}", bass, mix))

    condition = Condition(
        "stem-mix", "stem", (Channel("bass", Identity()), Channel("mix", Identity()))
    )
    per_track = condition.channel_paths(items, Cache(tmp_path / "cache"))

    assert len(per_track) == 3
    for item, (ch0, ch1) in zip(items, per_track, strict=True):
        assert len(  # two distinct cache entries, not one file listed twice
            {ch0, ch1}
        ) == 2
        assert ch0.name.startswith(item.sources().bass.stem)
        assert ch1.name.startswith(item.sources().mix.stem)


def test_the_stem_control_matches_its_bichannel_twins_track_filter():
    # "stem" keeps two-bass tracks (a stem is unambiguous whatever else sounds), so
    # it and "stem-mix" train on different corpora -- 33 tracks against 37 at
    # --limit 40. Comparing them would vary the input channels and the data at once.
    assert CONDITIONS["stem"].unambiguous_only is False
    assert CONDITIONS["stem-solo"].unambiguous_only is True
    assert CONDITIONS["stem-solo"].in_channels == 1
    assert (
        CONDITIONS["stem-solo"].unambiguous_only
        == CONDITIONS["stem-mix"].unambiguous_only
    )


def test_demucs_needs_no_control_twin():
    # It already filters to single-bass tracks, so demucs vs demucs-mix is
    # controlled as it stands -- the deployment-shaped comparison needs no extra tag.
    assert (
        CONDITIONS["demucs"].unambiguous_only
        == CONDITIONS["demucs-mix"].unambiguous_only
    )
    assert CONDITIONS["demucs"].source == CONDITIONS["demucs-mix"].source


def test_every_condition_tag_can_appear_in_a_metric_name():
    # A tag becomes a path segment in every metric it produces (note/<tag>/f1), and
    # MLflow rejects anything outside this set. A "+" separator got through review
    # once: training ran to completion and exited 0 while every validation metric
    # was refused, one epoch at a time, inside a callback that swallowed the error.
    import re

    safe = re.compile(r"^[A-Za-z0-9_.:-]+$")
    assert [t for t in CONDITIONS if not safe.match(t)] == []


def test_every_recipe_width_has_a_clean_condition_of_that_width():
    # This is the assertion that was missing. val_loader picked its clean condition
    # by an inline tag, a rename left it pointing at a "stem+mix" that no longer
    # existed, and nothing caught it -- building a val loader needs the corpus, so
    # no test reached that line. Checking the *table* needs no corpus at all.
    from stemify.training.conditions import CLEAN_BY_WIDTH

    for width, tag in CLEAN_BY_WIDTH.items():
        assert tag in CONDITIONS, f"clean condition {tag!r} for width {width} is gone"
        assert CONDITIONS[tag].in_channels == width


def test_a_clean_condition_exists_for_every_width_a_condition_can_have():
    # A condition whose width has no clean twin cannot be validated, and that only
    # surfaces after the training loader is already built.
    from stemify.training.conditions import CLEAN_BY_WIDTH

    widths = {c.in_channels for c in CONDITIONS.values()}
    assert widths <= set(CLEAN_BY_WIDTH), (
        f"no clean val condition for width(s) {sorted(widths - set(CLEAN_BY_WIDTH))}"
    )


def test_clean_val_conditions_all_filter_to_single_bass():
    # The pinned protocol (docs/ablation-plan.md): every reported condition is
    # unambiguous_only, so runs of different input widths early-stop against the same
    # track set. "stem" is the one condition that keeps multi-bass tracks, and using it
    # as the 1-channel clean condition would silently reintroduce the mismatch that
    # made tcn-v2's stem 0.878 incomparable to stem-solo's 0.9375.
    from stemify.training.conditions import CLEAN_BY_WIDTH

    for width, tag in CLEAN_BY_WIDTH.items():
        assert CONDITIONS[tag].unambiguous_only, (
            f"clean condition {tag!r} for width {width} keeps multi-bass tracks"
        )
