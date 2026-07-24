"""Cached validation scoring: given precomputed examples and a model, does it
produce sane note-level metrics. Uses a fake onset model and synthetic examples,
so no pyin or audio -- the OnsetValidator's own pyin caching is I/O, tested by
the smoke run, not here.
"""

import numpy as np
import torch
from torch import nn

from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.segmentation.learned import LearnedSegmenter
from stemify.training.cache import cached_f0
from stemify.training.validation import (
    ValExample,
    score_examples,
    score_voicing,
)

N_BINS = 8


def _segment(model):
    """The segment step the validator scores: wrap a (fake) model in a real
    LearnedSegmenter and hand back its cache-fed segment_features."""
    return LearnedSegmenter(model, device="cpu").segment_features


class _CountingBaseline:
    """Stands in for PyinBaseline: returns a fixed contour, counts computations."""

    name = "pyin"

    def __init__(self):
        self.calls = 0

    def predict_f0(self, path):
        self.calls += 1
        times = np.linspace(0, 1, 5)
        return Contour(times=times, freqs=np.full(5, 110.0),
                       confidence=np.full(5, 0.9))


def test_cached_f0_computes_once_then_reads_disk(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_text("x")
    cache = tmp_path / "contours"
    base = _CountingBaseline()

    first = cached_f0(base, audio, cache)
    assert base.calls == 1
    second = cached_f0(base, audio, cache)  # served from disk
    assert base.calls == 1
    np.testing.assert_array_equal(first.freqs, second.freqs)
    np.testing.assert_array_equal(first.confidence, second.confidence)
    assert list(cache.glob("*.partial.npz")) == []  # atomic write left no temp


def test_cached_f0_keys_on_path_so_conditions_dont_collide(tmp_path):
    # Two different audio files (stem vs separated) must get distinct cache entries.
    a, b = tmp_path / "stem.wav", tmp_path / "demucs.wav"
    a.write_text("a")
    b.write_text("b")
    cache = tmp_path / "contours"
    base = _CountingBaseline()
    cached_f0(base, a, cache)
    cached_f0(base, b, cache)
    assert base.calls == 2  # no collision -> both computed
    assert len(list(cache.glob("*.npz"))) == 2


def test_cached_f0_none_disables_cache(tmp_path):
    audio = tmp_path / "stem.wav"
    audio.write_text("x")
    base = _CountingBaseline()
    cached_f0(base, audio, None)
    cached_f0(base, audio, None)
    assert base.calls == 2  # recomputed each time, nothing written


class _FakeOnsetModel(nn.Module):
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
    ex = ValExample(frame_times=frame_times, contour=contour, notes=notes,
                    features=features)
    return ex, frame_times


def test_score_examples_returns_note_metrics():
    # A held pitch re-articulated at frames 0, 30, 60; the model fires there too,
    # so onsets split the held contour into the right notes.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    scores = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    assert set(scores) == {"note_f1", "note_precision", "note_recall", "fragmentation"}
    assert scores["note_recall"] > 0.5  # found the re-articulated notes
    assert 0.0 <= scores["note_f1"] <= 1.0


def test_missing_onsets_lower_recall():
    # Model fires only at frame 0 -> the two later re-articulations are missed.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    few = score_examples(_segment(_FakeOnsetModel([0])), [ex])
    many = score_examples(_segment(_FakeOnsetModel([0, 30, 60])), [ex])
    assert few["note_recall"] < many["note_recall"]


class _FakeVoicingModel(nn.Module):
    """A voicing head that ranks voiced frames (first half) above the rest."""

    def forward(self, x):  # (1, 1, F, T)
        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)
        voicing = torch.full((1, t, 1), -5.0)
        voicing[0, : t // 2, 0] = 5.0  # high P(voiced) on the first half
        return {"onset": onset, "voicing": voicing}


def test_score_voicing_returns_auroc_against_the_label():
    ex, _ = _example(40, hop=0.0116, held_pitch=45, onset_frames=[0])
    ex.voicing_target = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32)
    prob_fn = LearnedSegmenter(_FakeVoicingModel(), device="cpu").voicing_probs
    scores = score_voicing(prob_fn, [ex])
    assert scores["voicing_auroc"] == 1.0  # perfectly separates voiced from rest


def test_score_voicing_empty_for_onset_only_model():
    ex, _ = _example(40, hop=0.0116, held_pitch=45, onset_frames=[0])
    ex.voicing_target = np.concatenate([np.ones(20), np.zeros(20)]).astype(np.float32)
    prob_fn = LearnedSegmenter(_FakeOnsetModel([0]), device="cpu").voicing_probs
    assert score_voicing(prob_fn, [ex]) == {}  # no voicing head -> nothing to score