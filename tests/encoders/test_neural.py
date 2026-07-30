"""NeuralEncoder: the cached-features path the validator uses, and encode-from-audio.

Fake models drive known onset/voicing curves so the forward, Cues assembly, and the
decode that follows are exercised without a trained model.
"""

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from stemify.contour import Contour
from stemify.decoders.heuristic import HeuristicDecoder, onset_times_from_probs
from stemify.encoders.neural import NeuralEncoder
from stemify.features import SAMPLE_RATE, FrameFeatures
from stemify.sources import Sources


class _FakeOnsetModel(nn.Module):
    """Onset logits that peak at given fractions of the input length; no voicing."""

    def __init__(self, peak_fracs):
        super().__init__()
        self.peak_fracs = peak_fracs

    def forward(self, x):  # x: (1, 1, F, T)
        t = x.shape[-1]
        logits = torch.full((1, t, 1), -10.0)
        for frac in self.peak_fracs:
            logits[0, int(frac * t), 0] = 10.0
        return {"onset": logits}


class _FakeOnsetVoicingModel(nn.Module):
    """Onset never fires; voicing is 'sounding' only on the first half of frames."""

    def forward(self, x):  # x: (1, 1, F, T)
        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)
        voicing = torch.full((1, t, 1), -10.0)
        voicing[0, : t // 2, 0] = 10.0
        return {"onset": onset, "voicing": voicing}


def _held(n: int) -> tuple[Contour, FrameFeatures]:
    frame_times = np.arange(n) * 0.0116
    contour = Contour(frame_times, np.full(n, 110.0))  # voiced everywhere per pyin
    features = FrameFeatures(np.random.rand(8, n).astype(np.float32), frame_times)
    return contour, features


def test_encode_features_voicing_head_places_rests():
    # pyin tracks one held pitch, but the voicing head calls the second half a rest,
    # so the decoded note ends early -- the cached validator path end to end.
    n = 40
    contour, features = _held(n)
    cues = NeuralEncoder(_FakeOnsetVoicingModel(), device="cpu").encode_features(
        contour, features
    )
    assert cues.voicing is not None
    notes = HeuristicDecoder(merge_gap_s=None).decode(cues)
    assert len(notes) == 1
    assert notes[0].offset < features.frame_times[n // 2] + 0.05


def test_encode_features_without_voicing_falls_back_to_contour():
    n = 40
    contour, features = _held(n)
    cues = NeuralEncoder(_FakeOnsetModel([]), device="cpu").encode_features(
        contour, features
    )
    assert cues.voicing is None  # onset-only model
    assert len(HeuristicDecoder(merge_gap_s=None).decode(cues)) == 1


def test_encode_from_audio_peaks_the_onset_curve(tmp_path):
    wav = tmp_path / "tone.wav"
    dur_s = 1.0
    t = np.linspace(0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False)
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    cues = NeuralEncoder(_FakeOnsetModel([1 / 3, 2 / 3]), device="cpu").encode(
        Sources(bass=wav)
    )
    assert cues.onset.shape == cues.frame_times.shape
    assert cues.onset.max() > 0.99 and cues.onset.min() < 0.01
    onsets = onset_times_from_probs(cues.onset, cues.frame_times)
    assert len(onsets) == 2
    assert np.isclose(onsets[0], dur_s / 3, atol=0.05)
    assert np.isclose(onsets[1], 2 * dur_s / 3, atol=0.05)


# --------------------------------------------------------------------------- #
# mix-conditioned input: several stacked renderings of one track               #
# --------------------------------------------------------------------------- #
class _ChannelCountingModel(nn.Module):
    """Records the channel count it was handed; never fires an onset."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.seen = None

    def forward(self, x):  # x: (1, C, F, T)
        self.seen = tuple(x.shape[:2])
        return {"onset": torch.full((1, x.shape[-1], 1), -10.0)}


def test_encode_features_passes_a_stacked_input_through_unflattened():
    n = 12
    contour, _ = _held(n)
    model = _ChannelCountingModel(2)
    stacked = FrameFeatures(
        np.random.rand(2, 8, n).astype(np.float32), np.arange(n) * 0.0116
    )
    NeuralEncoder(model, device="cpu").encode_features(contour, stacked)
    assert model.seen == (1, 2)          # (batch, channels) -- the axis survives


def test_a_bare_cqt_still_reaches_a_single_channel_model():
    # (n_bins, T) gains the channel axis here, so every existing checkpoint and
    # cached feature array keeps working untouched.
    n = 12
    contour, features = _held(n)
    model = _ChannelCountingModel(1)
    NeuralEncoder(model, device="cpu").encode_features(contour, features)
    assert model.seen == (1, 1)


def test_encoding_the_wrong_channel_count_fails_loudly():
    # Silently feeding a 2-channel checkpoint one channel would produce cues, and
    # therefore notes, from a model reading half its input as absent.
    n = 12
    contour, features = _held(n)
    encoder = NeuralEncoder(_ChannelCountingModel(2), device="cpu")
    with pytest.raises(ValueError, match="expects 2 input channel"):
        encoder.encode_features(contour, features)


def test_encode_from_audio_reads_the_mix_for_a_second_channel(tmp_path):
    # At deployment `bass` is already whatever the caller separated, so both
    # channels are read as-is -- the training transforms have no role here.
    bass, mix = tmp_path / "bass.wav", tmp_path / "mix.wav"
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    sf.write(bass, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)
    sf.write(mix, 0.5 * np.sin(2 * np.pi * 220.0 * t), SAMPLE_RATE)

    model = _ChannelCountingModel(2)
    NeuralEncoder(model, device="cpu").encode(Sources(bass=bass, mix=mix))
    assert model.seen == (1, 2)


def test_a_mix_conditioned_model_needs_a_mix(tmp_path):
    bass = tmp_path / "bass.wav"
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    sf.write(bass, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    encoder = NeuralEncoder(_ChannelCountingModel(2), device="cpu")
    with pytest.raises(ValueError, match="no 'mix' audio"):
        encoder.encode(Sources(bass=bass))
