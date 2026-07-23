"""The ``Encoder`` interface: audio -> ``Cues``.

An encoder is the *cue-production* half of the note pipeline. It turns a track into
the per-frame evidence a decoder needs -- an onset curve and (optionally) a voicing
curve -- and bundles it with the pyin pitch contour into a ``Cues``. Two encoders,
one interface:

  - ``DSPEncoder`` -- spectral-flux onset envelope, no learning (the floor).
  - ``NeuralEncoder`` -- a trained backbone (TCN) + heads producing onset/voicing
    probabilities.

The pitch contour is passed *in* rather than computed here: pyin is the fixed pitch
tracker the project holds constant while onset/voicing are studied, and it is cached
independently of any checkpoint. So an encoder adds the *event* cues (onset,
voicing) to that pitch track -- it never re-derives pitch. Swapping ``DSPEncoder``
for ``NeuralEncoder`` is exactly the DSP-floor-vs-learned comparison, now a change
of one composed object rather than a subclass.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from stemify.contour import Contour
from stemify.cues import Cues


class Encoder(ABC):
    """Turns a track's audio (plus its pitch contour) into ``Cues``."""

    name: str

    @abstractmethod
    def encode(self, contour: Contour, audio_path: str | Path) -> Cues:
        """The ``Cues`` for a track: the given pitch ``contour`` bundled with the
        per-frame onset (and, where the encoder has it, voicing) evidence derived
        from the audio at ``audio_path``."""
