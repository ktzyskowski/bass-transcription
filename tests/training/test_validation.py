"""Cached validation scoring: given precomputed examples and a model, does it
produce sane note-level metrics. Uses fake models and synthetic examples, so no audio
-- the CheckpointValidator's own CQT caching is I/O, tested by the smoke run.
"""

import numpy as np
import pytest
import torch
from torch import nn

from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.music.notes import NoteEvent
from stemify.training.validation import (
    ValItem,
    score_cues_examples,
    score_examples,
)

N_BINS = 8


def _segment(model):
    """The segment step the validator scores: the LearnedEncoder + GreedyDecoder
    pair deployment composes, fed one ValItem -- the same path."""
    encoder = LearnedEncoder(model, device="cpu")
    decoder = GreedyDecoder()

    def segment(ex):
        return decoder.decode(encoder.encode_features(ex.load_features()))

    return segment


class _FakeFrameNet(nn.Module):
    """The parts of ``FrameNet``'s interface ``LearnedEncoder`` reads besides ``forward``
    -- declared, because the encoder no longer defaults them when a double omits them."""

    in_channels = 1
    frequency_heads: dict = {}

    HIDDEN = 8
    """Width of the stand-in trunk. Small and arbitrary: nothing here reads its values,
    only that a bundle from a model *with* a backbone carries one."""

    def heads_and_hidden(self, x):
        """The heads, plus a stand-in ``(B, T, HIDDEN)`` trunk.

        Zeros, because no double here has a backbone to produce a real representation --
        but the shape is honest, and returning it is what keeps a double from standing
        in for only half of ``FrameNet``. A scorer that pools the trunk will read zeros and
        learn nothing from them; that is a test writing its own inputs, not a silent
        production fallback."""
        out = self.forward(x)
        return out, torch.zeros(x.shape[0], x.shape[-1], self.HIDDEN)


class _FakeOnsetModel(_FakeFrameNet):
    """Onset logits peaking at given frame indices, over a flat held pitch.

    The pitch head is not decoration here: it is the only pitch source there is, so a
    model without one produces ``Frames`` a note decoder refuses by name. The pitch is
    held constant at the examples' own ``held_pitch``, so every note-level assertion
    turns purely on *where the boundaries fell* rather than on what was named.
    """

    frequency_heads = {"pitch": object()}

    def __init__(self, peak_frames, midi: float = 45.0):
        super().__init__()
        self.peak_frames = peak_frames
        self.midi = midi

    def forward(self, x):  # (1, 1, F, T)
        from stemify.audio.features import N_BINS as PITCH_BINS
        from stemify.models.heads import two_hot

        t = x.shape[-1]
        logits = torch.full((1, t, 1), -10.0)
        for f in self.peak_frames:
            logits[0, f, 0] = 10.0
        pitch = (two_hot(torch.full((1, t), self.midi), PITCH_BINS) - 0.5) * 40.0
        return {"onset": logits, "pitch": pitch}


def _example(n_frames, hop, held_pitch, onset_frames):
    frame_times = np.arange(n_frames) * hop
    features = np.random.rand(N_BINS, n_frames).astype(np.float32)  # (n_bins, T) CQT
    notes = [NoteEvent(onset=frame_times[f], offset=frame_times[f] + 0.2,
                       midi_pitch=held_pitch) for f in onset_frames]
    ex = ValItem(frame_times=frame_times, notes=notes, features=features)
    return ex, frame_times


def test_score_examples_returns_note_metrics():
    # A held pitch re-articulated at frames 0, 30, 60; the model fires there too,
    # so onsets split the held contour into the right notes.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    scores = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    # The metrics' own names, undecorated. Renaming to `note_*` here and back again
    # at the log site is what `NOTE_LOG_KEYS` replaced; the MLflow names are pinned
    # separately, in tests/metrics/test_logged_schema.py.
    assert set(scores) == {
        "tp", "fp", "fn", "precision", "recall", "f1",
        # Every false-positive bucket, not just fragmentation: precision alone
        # cannot say which error a change fixed, and they cost nothing extra --
        # the metric computes all four in one pass.
        "fragmentation", "interior_offpitch", "rest",
    }
    # The buckets partition the false positives, so they must add up. If they ever
    # stop doing so, some FP is being counted twice or dropped on the floor.
    assert scores["fp"] == (
        scores["fragmentation"] + scores["rest"] + scores["interior_offpitch"]
    )
    assert scores["recall"] > 0.5  # found the re-articulated notes
    assert 0.0 <= scores["f1"] <= 1.0


def test_missing_onsets_lower_recall():
    # Model fires only at frame 0 -> the two later re-articulations are missed.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    few = score_examples(_segment(_FakeOnsetModel([0])), [ex])
    many = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    assert few["recall"] < many["recall"]


class _FakeVoicingModel(_FakeFrameNet):
    """A voicing head that ranks voiced frames (first half) above the rest."""

    def forward(self, x):  # (1, 1, F, T)
        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)
        voicing = torch.full((1, t, 1), -5.0)
        voicing[0, : t // 2, 0] = 5.0  # high P(voiced) on the first half
        return {"onset": onset, "voicing": voicing}


def _encode(model):
    """The encode step the validator scores cues through: a checkpoint's
    LearnedEncoder over one ValItem."""
    encoder = LearnedEncoder(model, device="cpu")

    def encode(ex):
        return encoder.encode_features(ex.load_features())

    return encode


def test_score_cues_returns_voicing_auroc_against_the_label():
    ex, _ = _example(40, hop=0.0116, held_pitch=45, onset_frames=[0])
    ex.voicing_target = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32)
    results = {r.name: r.scalars for r in score_cues_examples(_encode(_FakeVoicingModel()), [ex])}
    assert results["voicing_auroc"]["voicing_auroc"] == 1.0  # separates voiced from rest
    assert "onset_pr" in results  # onset is always scored


def test_score_cues_no_voicing_channel_for_onset_only_model():
    ex, _ = _example(40, hop=0.0116, held_pitch=45, onset_frames=[0])
    ex.voicing_target = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32)
    names = {r.name for r in score_cues_examples(_encode(_FakeOnsetModel([0])), [ex])}
    assert "voicing_auroc" not in names  # no voicing head -> channel dropped
    assert "onset_pr" in names


# --------------------------------------------------------------------------- #
# The report has to say which pitch source produced its numbers                #
# --------------------------------------------------------------------------- #
def _framenet(pitch: bool):
    """Built through the recipe, not by hand: the pitch head is a *frequency* head
    living in its own collection (it taps the front-end before the temporal stack, so
    the frequency axis still exists), and hand-assembly gets that wrong silently --
    the encoder then reports "pyin" for a model that has a head."""
    from stemify.training.recipes import RECIPES

    return RECIPES["frame"](["clean"], pitch=pitch).build_model()


class _FakePitchHeadModel(_FakeFrameNet):
    """Onset logits plus a ``pitch`` frequency head, i.e. the shape every current
    checkpoint has. Declaring ``frequency_heads`` is what makes ``has_pitch_head``
    true -- the encoder reads that, not the weights."""

    frequency_heads = {"pitch": object()}

    def forward(self, x):  # (1, 1, F, T)
        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)
        onset[0, 0, 0] = 10.0
        voicing = torch.full((1, t, 1), 5.0)
        # Peaked on one bin so decode_contour has something unambiguous to read.
        pitch = torch.full((1, t, N_BINS), -10.0)
        pitch[0, :, N_BINS // 2] = 10.0
        return {"onset": onset, "voicing": voicing, "pitch": pitch}


def test_a_val_example_carries_no_pitch_source_of_its_own():
    """``ValItem`` used to hold a deferred pyin contour, plus a lazy property and a
    raising branch to manage it, for checkpoints with no pitch head. The contour comes
    out of the same forward pass as onset and voicing now, so an example is features
    and labels and nothing else."""
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0])
    assert not hasattr(ex, "contour")
    assert not hasattr(ex, "contour_source")

    scores = score_examples(_segment(_FakePitchHeadModel()), [ex])
    assert "f1" in scores


def test_a_model_with_no_pitch_head_cannot_be_decoded_to_notes():
    """The honest failure the pyin fallback used to hide. Such a checkpoint still
    scores at cues level -- its onset and voicing curves are real -- but nothing can
    name a note's pitch, so the decoder refuses rather than transcribing silence."""
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0])

    class _Headless(_FakeOnsetModel):
        frequency_heads: dict = {}

        def forward(self, x):
            return {"onset": super().forward(x)["onset"]}

    with pytest.raises(ValueError, match="carry none"):
        score_examples(_segment(_Headless([0])), [ex])
