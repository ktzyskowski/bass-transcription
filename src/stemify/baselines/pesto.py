"""PESTO baseline -- a learned, frame-level f0 estimator.

PESTO (SonyCSLParis) is a fast self-supervised pitch tracker. It is the learned
counterpart to pyin and the first real candidate for the pitch stage.

A caveat that matters enormously for bass: the default model, ``mir-1k_g7``, is
trained on *vocals* (the MIR-1K dataset), whose pitch range does not reach the
low bass register. In practice its predictions floor out around ~50 Hz, so bass
fundamentals below roughly G#1 are outside the model's range entirely -- and
where the fundamental is weak, it locks onto a harmonic instead (a probe on a
low F1 note came back an octave-plus-a-fifth high, i.e. the 3rd harmonic). The
RPA/RCA evaluation exists precisely to quantify how far this goes; expect the
octave-error rate (RCA minus RPA) to be large in the low register.

PESTO always emits a pitch, so voicing is derived by thresholding its
confidence output.
"""

from pathlib import Path

import librosa
import numpy as np
import torch

from stemify.baselines.base import F0Contour, PitchBaseline


class PestoBaseline(PitchBaseline):
    name = "pesto"

    def __init__(
        self,
        model_name: str = "mir-1k_g7",
        step_size: float = 10.0,  # ms between frames
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.step_size = step_size
        self.confidence_threshold = confidence_threshold

    def predict_f0(self, path: str | Path) -> F0Contour:
        import pesto  # deferred: importing pesto pulls in the torch model stack

        # Load at native sample rate to preserve the low-frequency content that
        # bass lives in; PESTO resamples internally as needed.
        audio, sr = librosa.load(path, sr=None, mono=True)
        x = torch.from_numpy(audio).float()
        timesteps, pitch, confidence, _activations = pesto.predict(
            x, sr, step_size=self.step_size, model_name=self.model_name
        )
        times = timesteps.detach().cpu().numpy() / 1000.0  # ms -> s
        pitch_hz = pitch.detach().cpu().numpy()
        confidence = confidence.detach().cpu().numpy()
        # Threshold confidence into voicing; unvoiced frames become 0 Hz.
        freqs = np.where(confidence >= self.confidence_threshold, pitch_hz, 0.0)
        return F0Contour(times=times, freqs=freqs, confidence=confidence)

    def predict_pitch(self, path: str | Path) -> float | None:
        contour = self.predict_f0(path)
        voiced = contour.freqs[contour.freqs > 0]
        if voiced.size == 0:
            return None
        return float(librosa.hz_to_midi(np.median(voiced)))