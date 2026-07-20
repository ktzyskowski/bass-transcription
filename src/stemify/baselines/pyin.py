"""pYIN baseline -- classic DSP monophonic pitch tracking, no learned model.

This is the floor. It adds no dependency beyond librosa and stands for "what
does a textbook pitch tracker get before we build anything." Anything learned
has to clear it to justify its own existence.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify import features
from stemify.baselines.base import PitchBaseline
from stemify.contour import Contour

# Search range. The lower bound is the open low E (E1, ~41 Hz). The upper bound
# sits an octave above the dataset's highest label (G#3) on purpose: it lets
# octave-up errors -- which weak bass fundamentals invite -- show up in the
# metrics instead of being silently clipped to the correct octave.
FMIN = librosa.note_to_hz("E1")
FMAX = librosa.note_to_hz("A4")
HOP_LENGTH = 512  # pyin's default; pinned so the time grid is explicit


class PyinBaseline(PitchBaseline):
    name = "pyin"

    def __init__(self, fmin: float = FMIN, fmax: float = FMAX) -> None:
        self.fmin = fmin
        self.fmax = fmax

    def predict_f0(self, path: str | Path) -> Contour:
        audio = features.load_audio(path)
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            audio,
            sr=features.SAMPLE_RATE,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=HOP_LENGTH,
        )
        times = librosa.times_like(
            f0, sr=features.SAMPLE_RATE, hop_length=HOP_LENGTH
        )
        # pyin marks unvoiced frames with NaN; mir_eval wants 0.0.
        freqs = np.nan_to_num(f0, nan=0.0)
        return Contour(times=times, freqs=freqs, confidence=voiced_prob)

    def predict_pitch(self, path: str | Path) -> float | None:
        contour = self.predict_f0(path)
        voiced = contour.freqs[contour.freqs > 0]
        if voiced.size == 0:
            return None
        # Median over voiced frames: robust to the pitch wobble at onset/decay.
        return float(librosa.hz_to_midi(np.median(voiced)))
