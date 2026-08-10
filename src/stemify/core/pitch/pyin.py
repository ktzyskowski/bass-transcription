"""``PyinPitch``: probabilistic YIN, the classic DSP monophonic pitch tracker.

The floor a learned head has to beat, and still the pitch source for any encoder
whose model has no pitch head.

Two structural weaknesses, both quantified in docs/experiments.md and both the reason
the learned head exists. It runs at hop 512 -- 23.2 ms, twice the CQT's frame period --
so every note boundary derived from it is quantised to half the 50 ms match tolerance.
And its pitch transition *lags* a real onset, which is what made a pitch change an
unusable cue for placing notes; the lag worsens on separated audio, where it also more
often fails to reach a note's pitch at all.

Torch-free, so the DSP path never pulls torch in through the pitch source.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.audio import features
from stemify.core.contour import Contour
from stemify.core.pitch.base import PitchEstimator

# Search range. The lower bound is the open low E (E1, ~41 Hz). The upper bound sits
# an octave above the dataset's highest label on purpose: it lets octave-up errors --
# which weak bass fundamentals invite -- show up instead of being clipped to the right
# octave.
# Distinct from ``features.FMIN`` (D#1), which is the CQT's lower *bin* edge and sits a
# semitone lower for margin. This is the lowest pitch pyin will search for.
# ``float(...)`` because ``note_to_hz`` returns a numpy scalar, and these are read as
# plain frequencies -- a default for a ``float`` parameter here, a divisor of a Tensor
# in ``features``. Converting at the definition beats converting at every use.
FMIN = float(librosa.note_to_hz("E1"))
FMAX = float(librosa.note_to_hz("A4"))
HOP_LENGTH = 512  # pyin's default; pinned so the time grid is explicit
# Twice ``features.HOP_LENGTH`` (256), which is the CQT's. Same name, different grid:
# any contour from this estimator is half the resolution of the model's frame grid, and
# every note boundary derived from it inherits that. See the note there.


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
