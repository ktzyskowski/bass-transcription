"""LearnedEncoder: the cached-features path the validator uses, and encode-from-audio.

Fake models drive known onset/voicing curves so the forward, Frames assembly, and the
decode that follows are exercised without a trained model.
"""

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from stemify.music.contour import Contour
from stemify.transcription.decoders.greedy import GreedyDecoder, onset_times_from_probs
from stemify.models.learned_encoder import LearnedEncoder
from stemify.audio.features import SAMPLE_RATE, FrameFeatures
from stemify.audio.sources import Sources


class _FakeFrameNet(nn.Module):
    """Base for the doubles below: the parts of ``FrameNet``'s interface that
    ``LearnedEncoder`` reads besides ``forward``.

    Declared rather than left off. The encoder used to reach for these with
    ``getattr(..., default)``, which let a double stand in while implementing only
    half the interface -- and the default it fell back to was a plausible wrong
    answer in production, not just in tests. A double should be honest about the
    shape it is standing in for."""

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


class _FakeOnsetVoicingModel(_FakeFrameNet):
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
    cues = LearnedEncoder(_FakeOnsetVoicingModel(), device="cpu").encode_features(
        contour, features
    )
    assert cues.voicing is not None
    notes = GreedyDecoder(merge_gap_s=None).decode(cues)
    assert len(notes) == 1
    assert notes[0].offset < features.frame_times[n // 2] + 0.05


def test_encode_features_without_voicing_falls_back_to_contour():
    n = 40
    contour, features = _held(n)
    cues = LearnedEncoder(_FakeOnsetModel([]), device="cpu").encode_features(
        contour, features
    )
    assert cues.voicing is None  # onset-only model
    assert len(GreedyDecoder(merge_gap_s=None).decode(cues)) == 1


def test_encode_from_audio_peaks_the_onset_curve(tmp_path):
    wav = tmp_path / "tone.wav"
    dur_s = 1.0
    t = np.linspace(0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False)
    sf.write(wav, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    cues = LearnedEncoder(_FakeOnsetModel([1 / 3, 2 / 3]), device="cpu").encode(
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
class _ChannelCountingModel(_FakeFrameNet):
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
    LearnedEncoder(model, device="cpu").encode_features(contour, stacked)
    assert model.seen == (1, 2)          # (batch, channels) -- the axis survives


def test_a_bare_cqt_still_reaches_a_single_channel_model():
    # (n_bins, T) gains the channel axis here, so every existing checkpoint and
    # cached feature array keeps working untouched.
    n = 12
    contour, features = _held(n)
    model = _ChannelCountingModel(1)
    LearnedEncoder(model, device="cpu").encode_features(contour, features)
    assert model.seen == (1, 1)


def test_encoding_the_wrong_channel_count_fails_loudly():
    # Silently feeding a 2-channel checkpoint one channel would produce cues, and
    # therefore notes, from a model reading half its input as absent.
    n = 12
    contour, features = _held(n)
    encoder = LearnedEncoder(_ChannelCountingModel(2), device="cpu")
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
    LearnedEncoder(model, device="cpu").encode(Sources(bass=bass, mix=mix))
    assert model.seen == (1, 2)


def test_a_mix_conditioned_model_needs_a_mix(tmp_path):
    bass = tmp_path / "bass.wav"
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    sf.write(bass, 0.5 * np.sin(2 * np.pi * 110.0 * t), SAMPLE_RATE)

    encoder = LearnedEncoder(_ChannelCountingModel(2), device="cpu")
    with pytest.raises(ValueError, match="no 'mix' audio"):
        encoder.encode(Sources(bass=bass))


# --------------------------------------------------------------------------- #
# the learned contour: a model with a pitch head is its own pitch source       #
# --------------------------------------------------------------------------- #
class _FakePitchModel(_FakeFrameNet):
    """Confidently predicts one MIDI pitch everywhere; voicing on the first half."""

    frequency_heads = {"pitch": object()}   # what has_pitch_head reads

    def __init__(self, midi: float = 45.0):
        super().__init__()
        self.midi = midi

    def forward(self, x):  # x: (1, 1, F, T)
        from stemify.audio.features import N_BINS
        from stemify.models.heads import two_hot

        t = x.shape[-1]
        onset = torch.full((1, t, 1), -10.0)
        voicing = torch.full((1, t, 1), -10.0)
        voicing[0, : t // 2, 0] = 10.0
        target = two_hot(torch.full((1, t), self.midi), N_BINS)   # (1, t, n_bins)
        return {"onset": onset, "voicing": voicing, "pitch": (target - 0.5) * 40.0}


def _flat_features(t: int = 40) -> FrameFeatures:
    from stemify.audio.features import N_BINS

    return FrameFeatures(np.zeros((N_BINS, t), dtype=np.float32), np.arange(t) * 0.01)


def test_pitch_head_replaces_the_supplied_contour():
    """The head is the pitch source when the model has one -- the caller's cached pyin
    contour is exactly what such a run exists to stop using."""
    import librosa

    feats = _flat_features()
    pyin_says = Contour(feats.frame_times, np.full(feats.frame_times.size, 200.0))
    enc = LearnedEncoder(_FakePitchModel(45.0), device="cpu")

    assert enc.has_pitch_head and enc.pitch_source == "learned"
    cues = enc.encode_features(pyin_says, feats)
    # Voiced half carries the head's pitch, not pyin's 200 Hz.
    voiced = cues.contour.freqs[: feats.frame_times.size // 2]
    assert np.allclose(librosa.hz_to_midi(voiced), 45.0, atol=0.05)
    # The contour lands on the FEATURE grid -- the point of the swap.
    assert np.array_equal(cues.contour.times, feats.frame_times)


def test_voicing_head_gates_the_learned_contour():
    feats = _flat_features()
    enc = LearnedEncoder(_FakePitchModel(45.0), device="cpu")
    cues = enc.encode_features(Contour(feats.frame_times, np.zeros(40)), feats)
    half = feats.frame_times.size // 2
    assert (cues.contour.freqs[:half] > 0).all()     # voiced -> a pitch
    assert (cues.contour.freqs[half:] == 0.0).all()  # rest -> no pitch


def test_use_pyin_keeps_the_supplied_contour():
    """The A/B escape hatch: same checkpoint, the other pitch source."""
    feats = _flat_features()
    pyin_says = Contour(feats.frame_times, np.full(feats.frame_times.size, 200.0))
    enc = LearnedEncoder(_FakePitchModel(45.0), device="cpu", use_pyin=True)

    assert enc.has_pitch_head and enc.pitch_source == "pyin"
    cues = enc.encode_features(pyin_says, feats)
    assert np.allclose(cues.contour.freqs, 200.0)


def test_a_model_without_a_pitch_head_still_uses_the_supplied_contour():
    feats = _flat_features()
    supplied = Contour(feats.frame_times, np.full(feats.frame_times.size, 200.0))
    enc = LearnedEncoder(_FakeOnsetVoicingModel(), device="cpu")

    assert not enc.has_pitch_head and enc.pitch_source == "pyin"
    assert np.allclose(enc.encode_features(supplied, feats).contour.freqs, 200.0)
