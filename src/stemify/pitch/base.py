"""``PitchEstimator``: audio -> a ``Contour``. The pipeline's swappable pitch source.

The interface is deliberately *audio in*, not features in. That is what makes the
implementations substitutable at all: pyin needs a waveform, a learned head needs a
CQT, and the only input both can be asked for is the file. An estimator that took
features would be usable by one caller and not the other.

The cost of that choice is real and worth naming. ``NeuralEncoder`` has already pushed
a track's CQT through the backbone by the time it wants a contour, so going back out to
``estimate(audio_path)`` would recompute the CQT and the forward pass it just did. It
therefore does *not* use a ``PitchEstimator`` when its model carries a pitch head -- it
calls ``models.heads.decode_contour`` on logits it already has. ``NeuralPitch`` calls
the same function, so the two entry points share one decode rather than drifting; what
differs is only how the logits were obtained.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from stemify.contour import Contour


class PitchEstimator(ABC):
    """Estimates a pitch contour from audio -- the pipeline's swappable pitch source."""

    name: str

    @abstractmethod
    def estimate(self, audio_path: str | Path) -> Contour:
        """The pitch contour for ``audio_path`` (``freqs == 0.0`` where unvoiced)."""
