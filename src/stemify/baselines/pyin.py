"""pYIN baseline -- classic DSP monophonic pitch tracking, no learned model.

This is the floor. It adds no dependency beyond librosa and stands for "what
does a textbook pitch tracker get before we build anything." Anything learned
has to clear it to justify its own existence.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.data import features

# Search range. The lower bound is the open low E (E1, ~41 Hz). The upper bound
# sits an octave above the dataset's highest label (G#3) on purpose: it lets
# octave-up errors -- which weak bass fundamentals invite -- show up in the
# metrics instead of being silently clipped to the correct octave.
FMIN = librosa.note_to_hz("E1")
FMAX = librosa.note_to_hz("A4")


class PyinBaseline(PitchBaseline):
    name = "pyin"

    def __init__(self, fmin: float = FMIN, fmax: float = FMAX) -> None:
        self.fmin = fmin
        self.fmax = fmax

    def predict_pitch(self, path: str | Path) -> float | None:
        audio = features.load_audio(path)
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            audio, sr=features.SAMPLE_RATE, fmin=self.fmin, fmax=self.fmax
        )
        voiced = f0[~np.isnan(f0)]
        if voiced.size == 0:
            return None
        # Median over voiced frames: robust to the pitch wobble at onset/decay.
        return float(librosa.hz_to_midi(np.median(voiced)))
