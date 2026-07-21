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
from stemify.segmentation import HeuristicSegmenter
from stemify.training.validation import ValExample, score_examples

N_BINS = 8


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
    features = torch.randn(1, 1, N_BINS, n_frames)
    notes = [NoteEvent(onset=frame_times[f], offset=frame_times[f] + 0.2,
                       midi_pitch=held_pitch) for f in onset_frames]
    return ValExample(features, frame_times, contour, notes), frame_times


def test_score_examples_returns_note_metrics():
    # A held pitch re-articulated at frames 0, 30, 60; the model fires there too,
    # so onsets split the held contour into the right notes.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    model = _FakeOnsetModel([0, 30, 60])
    scores = score_examples(
        model, [ex], HeuristicSegmenter(), threshold=0.5, min_gap_s=0.03,
    )
    assert set(scores) == {"note_f1", "note_precision", "note_recall", "fragmentation"}
    assert scores["note_recall"] > 0.5  # found the re-articulated notes
    assert 0.0 <= scores["note_f1"] <= 1.0


def test_missing_onsets_lower_recall():
    # Model fires only at frame 0 -> the two later re-articulations are missed.
    ex, _ = _example(90, hop=0.0116, held_pitch=45, onset_frames=[0, 30, 60])
    few = score_examples(_FakeOnsetModel([0]), [ex], HeuristicSegmenter(), 0.5, 0.03)
    many = score_examples(_FakeOnsetModel([0, 30, 60]), [ex], HeuristicSegmenter(), 0.5, 0.03)
    assert few["note_recall"] < many["note_recall"]