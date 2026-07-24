"""Pitch estimation: audio -> a ``Contour``. The encoder's pitch source, swappable.

An ``Encoder`` produces the pitch contour that anchors each note's pitch. Today that
is pyin (classic DSP); a learned pitch head will replace it later. Both are
``PitchEstimator``s, so an encoder holds one behind this interface and the head drops
in with no change to the encoder -- the "encoder produces the contour" promise made
concrete.

This is deliberately distinct from ``baselines/``: a ``PitchEstimator`` is an
*internal component* of the pipeline (pyin as the contour source inside
``DSPEncoder``/``NeuralEncoder``), whereas a ``PitchBaseline`` is a whole system the
evaluation harness scores. ``PyinBaseline`` now wraps ``PyinPitch`` rather than
reimplementing it, so the DSP lives in one place and the two roles stay separate.

Onset and voicing get no such ABC: they are *outputs* of the ``FrameTagger``'s heads,
not pluggable strategies with two implementations, so abstracting them would be
ceremony around a tensor.

Kept a single torch-free module (pyin only) so the light DSP path never pulls torch;
promote to a ``pitch/`` package (base + pyin + neural) when the learned head lands.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import librosa
import numpy as np

from stemify import features
from stemify.contour import Contour

# Search range. The lower bound is the open low E (E1, ~41 Hz). The upper bound sits
# an octave above the dataset's highest label on purpose: it lets octave-up errors --
# which weak bass fundamentals invite -- show up instead of being clipped to the right
# octave.
FMIN = librosa.note_to_hz("E1")
FMAX = librosa.note_to_hz("A4")
HOP_LENGTH = 512  # pyin's default; pinned so the time grid is explicit


class PitchEstimator(ABC):
    """Estimates a pitch contour from audio -- the pipeline's swappable pitch source."""

    name: str

    @abstractmethod
    def estimate(self, audio_path: str | Path) -> Contour:
        """The pitch contour for ``audio_path`` (``freqs == 0.0`` where unvoiced)."""


class PyinPitch(PitchEstimator):
    """pyin (classic DSP monophonic tracking) -- the floor a learned head must beat."""

    name = "pyin"

    def __init__(self, fmin: float = FMIN, fmax: float = FMAX) -> None:
        self.fmin = fmin
        self.fmax = fmax

    def estimate(self, audio_path: str | Path) -> Contour:
        audio = features.load_audio(audio_path)
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            audio,
            sr=features.SAMPLE_RATE,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=HOP_LENGTH,
        )
        times = librosa.times_like(f0, sr=features.SAMPLE_RATE, hop_length=HOP_LENGTH)
        # pyin marks unvoiced frames with NaN; mir_eval (and Contour) want 0.0.
        freqs = np.nan_to_num(f0, nan=0.0)
        return Contour(times=times, freqs=freqs, confidence=voiced_prob)
