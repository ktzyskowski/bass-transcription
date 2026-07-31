"""``PyinPitch``: probabilistic YIN, the classic DSP monophonic pitch tracker.

The floor a learned head has to beat, and still the pitch source for any encoder
whose model has no pitch head.

Its two known weaknesses are both measured (docs/experiments.md, 2026-07-31) and both
are why the learned head exists. It runs at hop 512 -- 23.2 ms, twice the CQT's frame
period -- so every note boundary derived from it is quantised to half the 50 ms match
tolerance. And its transition lags a real onset by a median 23 ms (37.5 ms mean on
separated audio), which is what made a pitch change an unusable cue for placing notes.
On separated audio it fails to reach a note's pitch at all on 12.5% of pitch changes,
against 1.1% on clean stems.

Torch-free, so the DSP path never pulls torch in through the pitch source.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify import features
from stemify.contour import Contour
from stemify.pitch.base import PitchEstimator

# Search range. The lower bound is the open low E (E1, ~41 Hz). The upper bound sits
# an octave above the dataset's highest label on purpose: it lets octave-up errors --
# which weak bass fundamentals invite -- show up instead of being clipped to the right
# octave.
FMIN = librosa.note_to_hz("E1")
FMAX = librosa.note_to_hz("A4")
HOP_LENGTH = 512  # pyin's default; pinned so the time grid is explicit


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
