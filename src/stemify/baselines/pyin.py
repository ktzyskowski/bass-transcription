"""pYIN baseline -- classic DSP monophonic pitch tracking, no learned model.

This is the floor. It adds no dependency beyond librosa and stands for "what
does a textbook pitch tracker get before we build anything." Anything learned
has to clear it to justify its own existence.

The pitch tracking itself lives in ``stemify.pitch.PyinPitch`` (the pipeline's
contour source); this wraps it in the ``PitchBaseline`` interface the eval harness
scores, so the DSP is defined once and the two roles -- internal component vs. scored
system -- stay separate.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.contour import Contour
from stemify.pitch import FMAX, FMIN, PyinPitch


class PyinBaseline(PitchBaseline):
    name = "pyin"

    def __init__(self, fmin: float = FMIN, fmax: float = FMAX) -> None:
        self._pitch = PyinPitch(fmin, fmax)

    def predict_f0(self, path: str | Path) -> Contour:
        return self._pitch.estimate(path)

    def predict_pitch(self, path: str | Path) -> float | None:
        contour = self.predict_f0(path)
        voiced = contour.freqs[contour.freqs > 0]
        if voiced.size == 0:
            return None
        # Median over voiced frames: robust to the pitch wobble at onset/decay.
        return float(librosa.hz_to_midi(np.median(voiced)))
