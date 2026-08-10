"""``ClassicalEncoder``: the DSP-floor cue producer (no learning).

Pitch from pyin, onsets from spectral flux, no voicing head -- the baseline a
learned encoder must beat. The onset channel of the ``Frames`` it returns is a
per-frame indicator (a spike at each flux-detected onset, on the contour grid),
which the ``GreedyDecoder`` peak-picks back to onset times; voicing is left
``None`` so the decoder falls back to the contour's own ``freqs > 0``. Expressed in
the new pipeline the old DSP floor is ``Transcriber(ClassicalEncoder(), GreedyDecoder())``.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.transcription.frames import Frames
from stemify.transcription.encoders.base import Encoder
from stemify.transcription.pitch import PitchEstimator, PyinPitch
from stemify.audio.sources import Sources

# Peak-picking threshold for onset detection: an onset must rise this far above
# the local mean flux to fire. librosa's default is 0.07. Raising it drops the
# weak flux peaks -- mid-sustain bow/finger noise -- that spuriously split a held
# note, while keeping the strong attacks of real re-plucks, because spurious
# onsets are systematically weaker than real ones. A sweep on Slakh stems put the
# F1 optimum near 0.20 (F1 71% -> 73%, precision +7 pts at -6 pts recall), which
# strictly beats the merge pass. Left at None (librosa default) because that
# optimum is tuned on isolated stems; the separated/mixed deployment condition has
# a different flux distribution and needs its own validation before the default
# moves. Unlike a scalar threshold, a learned onset head sees the whole attack
# shape, so this is the ceiling it should beat.
ONSET_DELTA: float | None = None


def detect_onsets(
    path: str | Path, sr: int = 22050, delta: float | None = ONSET_DELTA
) -> np.ndarray:
    """Onset times (seconds) via spectral flux, backtracked to the energy rise.

    ``backtrack`` moves each detection to the preceding local energy minimum, so the
    reported time is the note's actual start rather than the flux peak a few frames
    later. ``delta`` is the peak-picking threshold; None uses librosa's default.
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    # Branched rather than splatting a ``{}``-or-``{"delta": ...}`` dict. "None means
    # librosa's own default" is the actual rule, and this states it; the dict encoded it
    # as an absent key, and a ``**dict[str, float]`` splat is also checked against every
    # keyword the callee has -- including ones typed ``bool`` and ``int``.
    onsets: np.ndarray = (
        librosa.onset.onset_detect(y=audio, sr=sr, units="time", backtrack=True)
        if delta is None
        else librosa.onset.onset_detect(
            y=audio, sr=sr, units="time", backtrack=True, delta=delta
        )
    )
    return onsets


class ClassicalEncoder(Encoder):
    """DSP cue producer: pyin pitch + spectral-flux onsets, no voicing head."""

    name = "dsp"

    def __init__(
        self,
        pitch: PitchEstimator | None = None,
        onset_delta: float | None = ONSET_DELTA,
        use_onsets: bool = True,
    ) -> None:
        self.pitch = pitch or PyinPitch()
        self.onset_delta = onset_delta
        # Onsets are the only cue for a same-pitch re-articulation; off, those notes
        # merge -- useful when measuring how much onset detection contributes.
        self.use_onsets = use_onsets

    def encode(self, sources: Sources) -> Frames:
        contour = self.pitch.estimate(sources.bass)
        onset = np.zeros(contour.times.shape, dtype=np.float32)
        if self.use_onsets and contour.times.size:
            times = detect_onsets(sources.bass, delta=self.onset_delta)
            j = np.searchsorted(contour.times, times)
            j = j[(j >= 0) & (j < contour.times.size)]
            onset[j] = 1.0  # a spike per onset; the decoder peak-picks it back
        # No voicing head, so the key is absent rather than present-and-None: "which
        # heads exist" is a fact about the producer, and a decoder asks by membership.
        return Frames(
            heads={"onset": onset}, frame_times=contour.times, contour=contour
        )
