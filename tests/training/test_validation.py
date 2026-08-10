"""Cached validation scoring: given precomputed examples and a model, does it
produce sane note-level metrics. Uses a fake onset model and synthetic examples,
so no pyin or audio -- the CheckpointValidator's own pyin caching is I/O, tested by
the smoke run, not here.
"""

import numpy as np
import torch
from torch import nn

from stemify.core.contour import Contour
from stemify.core.decoders.greedy import GreedyDecoder
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
    needs_contour = encoder.pitch_source == "pyin"

    def segment(ex):
        contour = ex.contour if needs_contour else None
        return decoder.decode(encoder.encode_features(contour, ex.load_features()))

    return segment


class _CountingBaseline:
    """Stands in for a PitchEstimator: returns a fixed contour, counts calls."""

    name = "pyin"

    def __init__(self):
        self.calls = 0

    def estimate(self, path):
        self.calls += 1
        times = np.linspace(0, 1, 5)
        return Contour(times=times, freqs=np.full(5, 110.0),
                       confidence=np.full(5, 0.9))


class _FakeCueNet(nn.Module):
    """The parts of ``CueNet``'s interface ``LearnedEncoder`` reads besides ``forward``
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
        in for only half of ``CueNet``. A scorer that pools the trunk will read zeros and
        learn nothing from them; that is a test writing its own inputs, not a silent
        production fallback."""
        out = self.forward(x)
        return out, torch.zeros(x.shape[0], x.shape[-1], self.HIDDEN)


class _FakeOnsetModel(_FakeCueNet):
    """Onset logits peaking at given frame indices."""

    def __init__(self, peak_frames):
        super().__init__()
        self.peak_frames = peak_frames

    def forward(self, x):  # (1, 1, F, T)
        t = x.shape[-1]
        logits = torch.full((1, t, 1), -10.0)
        for f in self.peak_frames:
            logits[0, f, 0] = 10.0
        return {"onset": logits}


def _example(n_frames, hop, held_pitch, onset_frames):
    frame_times = np.arange(n_frames) * hop
    # a single held note across the whole clip at held_pitch
    freqs = np.full(n_frames, 440.0 * 2 ** ((held_pitch - 69) / 12))
    contour = Contour(frame_times, freqs)
    features = np.random.rand(N_BINS, n_frames).astype(np.float32)  # (n_bins, T) CQT
    notes = [NoteEvent(onset=frame_times[f], offset=frame_times[f] + 0.2,
                       midi_pitch=held_pitch) for f in onset_frames]
    ex = ValItem(frame_times=frame_times, contour_source=lambda: contour,
                    notes=notes, features=features)
    return ex, frame_times


def test_score_examples_returns_note_metrics():
    # A held pitch re-articulated at frames 0, 30, 60; the model fires there too,
    # so onsets split the held contour into the right notes.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    scores = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    assert set(scores) == {
        "note_f1", "note_precision", "note_recall",
        # Every false-positive bucket, not just fragmentation: precision alone
        # cannot say which error a change fixed, and they cost nothing extra --
        # the metric computes all four in one pass.
        "fragmentation", "fp", "fp_rest", "fp_interior_offpitch",
    }
    # The buckets partition the false positives, so they must add up. If they ever
    # stop doing so, some FP is being counted twice or dropped on the floor.
    assert scores["fp"] == (
        scores["fragmentation"] + scores["fp_rest"] + scores["fp_interior_offpitch"]
    )
    assert scores["note_recall"] > 0.5  # found the re-articulated notes
    assert 0.0 <= scores["note_f1"] <= 1.0


def test_missing_onsets_lower_recall():
    # Model fires only at frame 0 -> the two later re-articulations are missed.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    few = score_examples(_segment(_FakeOnsetModel([0])), [ex])
    many = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    assert few["note_recall"] < many["note_recall"]


class _FakeVoicingModel(_FakeCueNet):
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
    needs_contour = encoder.pitch_source == "pyin"

    def encode(ex):
        contour = ex.contour if needs_contour else None
        return encoder.encode_features(contour, ex.load_features())

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
def _cuenet(pitch: bool):
    """Built through the recipe, not by hand: the pitch head is a *frequency* head
    living in its own collection (it taps the front-end before the temporal stack, so
    the frequency axis still exists), and hand-assembly gets that wrong silently --
    the encoder then reports "pyin" for a model that has a head."""
    from stemify.training.recipes import RECIPES

    return RECIPES["frame"](["clean"], pitch=pitch).build_model()


def test_the_encoder_reports_which_pitch_source_it_used():
    assert LearnedEncoder(_cuenet(pitch=True), device="cpu").pitch_source == "learned"
    assert LearnedEncoder(_cuenet(pitch=False), device="cpu").pitch_source == "pyin"
    # ...and asking for pyin explicitly overrides a head that exists.
    forced = LearnedEncoder(_cuenet(pitch=True), device="cpu", use_pyin=True)
    assert forced.pitch_source == "pyin"


def test_the_validation_report_records_the_pitch_source(tmp_path):
    """A learned-pitch report and a pyin report are otherwise identical in shape, for
    a choice worth ~5 F1. Unrecorded, a saved artifact cannot say which made it."""
    import json

    from stemify.training.validation import build_eval_fn

    def report_for(pitch: bool) -> dict:
        out = tmp_path / ("learned" if pitch else "pyin")
        model = _cuenet(pitch)
        fn = build_eval_fn({}, report_dir=out, log_artifact=lambda p, e: None,
                      build_model=lambda: _cuenet(pitch))
        fn({"model": model.state_dict(), "loss": {}}, 0)
        return json.loads((out / "validation_epoch000.json").read_text())

    assert report_for(True)["pitch_source"] == "learned"
    assert report_for(False)["pitch_source"] == "pyin"


class _FakePitchHeadModel(_FakeCueNet):
    """Onset logits plus a ``pitch`` frequency head, i.e. the shape every current
    checkpoint has. Declaring ``frequency_heads`` is what makes ``pitch_source``
    report "learned" -- the encoder reads that, not the weights."""

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


def test_pitch_head_model_never_reaches_for_pyin():
    """The point of deferring the contour: a checkpoint that names its own pitch must
    not cause a DSP contour to be computed.

    It used to. ``CheckpointValidator`` ran pyin over every val track at construction,
    and ``encode_features`` overwrote the result on the next line for any model with a
    ``FrequencyHead`` -- a full pyin pass per track, discarded. This asserts on the
    *source* rather than on timing, so it fails if anything starts reading the contour
    again regardless of how fast pyin happens to be.
    """
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0])

    def explode() -> Contour:
        raise AssertionError(
            "a pitch-head checkpoint asked for a pyin contour; the head already "
            "supplies one and it would be discarded"
        )

    ex.contour_source = explode
    ex._contour = None

    scores = score_examples(_segment(_FakePitchHeadModel()), [ex])
    assert "note_f1" in scores


def test_headless_model_still_gets_its_contour():
    """The other half: a model with no pitch head has no other pitch source, so the
    deferred contour must still be reachable -- five checkpoints on disk are this
    shape. Counts calls to pin that it is fetched once, not per score."""
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0])
    calls = []
    contour = ex.contour  # the fixture's contour, before we wrap the source
    ex._contour = None
    ex.contour_source = lambda: (calls.append(1), contour)[1]

    segment = _segment(_FakeOnsetModel([0]))
    segment(ex)
    segment(ex)
    assert calls == [1], "the contour should be computed once and memoised"
