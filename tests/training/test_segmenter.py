"""LearnedSegmenter: peak-picking (pure) and the whole audio -> onsets path.

The integration test writes a short wav and drives a fake onset model whose
probability curve peaks at known fractions of the track, so the CQT length,
model forward, peak-pick, and time conversion are all exercised end to end
without a trained model.
"""

import numpy as np
import soundfile as sf
import torch
from torch import nn

from stemify.contour import Contour
from stemify.features import SAMPLE_RATE, Features
from stemify.segmentation.learned import LearnedSegmenter, onset_times_from_probs


def test_peak_pick_finds_maxima_above_threshold():
    frame_times = np.arange(20) * 0.01
    probs = np.zeros(20)
    probs[[5, 12]] = 0.9  # two clear peaks
    probs[8] = 0.3        # below threshold
    times = onset_times_from_probs(probs, frame_times, threshold=0.5, min_gap_s=0.02)
    assert np.allclose(times, [0.05, 0.12])


def test_peak_pick_enforces_min_gap():
    frame_times = np.arange(20) * 0.01
    probs = np.zeros(20)
    probs[10] = 0.9
    probs[11] = 0.8  # adjacent, within the refractory gap -> suppressed
    times = onset_times_from_probs(probs, frame_times, threshold=0.5, min_gap_s=0.05)
    assert len(times) == 1
    assert np.isclose(times[0], 0.10)


def test_peak_pick_empty():
    assert onset_times_from_probs(np.zeros(0), np.zeros(0)).size == 0


class _FakeOnsetModel(nn.Module):
    """Onset logits that peak at given fractions of the input length."""

    def __init__(self, peak_fracs):
        super().__init__()
        self.peak_fracs = peak_fracs

    def forward(self, x):  # x: (1, 1, F, T)
        t = x.shape[-1]
        logits = torch.full((1, t, 1), -10.0)
        for frac in self.peak_fracs:
            logits[0, int(frac * t), 0] = 10.0
        return {"onset": logits}


def test_onset_probs_and_times_from_audio(tmp_path):
    wav = tmp_path / "tone.wav"
    dur_s = 1.0
    t = np.linspace(0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False)
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    seg = LearnedSegmenter(_FakeOnsetModel([1 / 3, 2 / 3]), device="cpu")
    probs, frame_times = seg.onset_probs(wav)
    assert probs.shape == frame_times.shape          # dense per-frame curve
    assert probs.max() > 0.99 and probs.min() < 0.01  # sigmoid of the +/-10 logits

    onsets = seg._onset_times(wav)
    assert len(onsets) == 2
    # peaks at 1/3 and 2/3 of the track
    assert np.isclose(onsets[0], dur_s / 3, atol=0.05)
    assert np.isclose(onsets[1], 2 * dur_s / 3, atol=0.05)


class _FakeOnsetVoicingModel(nn.Module):
    """Onset never fires; voicing is 'sounding' only on the first half of frames --
    a model whose voicing head, not its onsets, decides the notes."""

    def forward(self, x):  # x: (1, 1, F, T)
        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)          # no onsets
        voicing = torch.full((1, t, 1), -10.0)
        voicing[0, : t // 2, 0] = 10.0                 # voiced first half only
        return {"onset": onset, "voicing": voicing}


def test_segment_features_uses_the_voicing_head_to_place_rests():
    # pyin tracks one held pitch across the whole clip (would be a single note), but
    # the model's voicing head calls the second half a rest, so the note ends early.
    n = 40
    frame_times = np.arange(n) * 0.0116
    contour = Contour(frame_times, np.full(n, 110.0))  # voiced everywhere per pyin
    features = Features(np.random.rand(8, n).astype(np.float32), frame_times)

    seg = LearnedSegmenter(_FakeOnsetVoicingModel(), device="cpu", merge_gap_s=None)
    notes = seg.segment_features(contour, features)

    assert len(notes) == 1                     # the learned rest bounded it to one note
    assert notes[0].offset < frame_times[n // 2] + 0.05  # ends at the voiced/rest edge


def test_segment_features_without_voicing_head_falls_back_to_contour():
    # An onset-only model supplies no voicing, so grouping uses pyin's own voicing
    # (freqs > 0): the whole voiced clip is one note.
    n = 40
    frame_times = np.arange(n) * 0.0116
    contour = Contour(frame_times, np.full(n, 110.0))
    features = Features(np.random.rand(8, n).astype(np.float32), frame_times)

    seg = LearnedSegmenter(_FakeOnsetModel([]), device="cpu", merge_gap_s=None)
    assert len(seg.segment_features(contour, features)) == 1