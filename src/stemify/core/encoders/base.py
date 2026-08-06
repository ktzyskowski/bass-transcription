"""The ``Encoder`` interface: audio (``Sources``) -> ``Cues``.

An encoder is the *cue-production* half of the note pipeline. It turns a track's
audio into all the per-frame evidence a decoder needs -- an onset curve, a voicing
curve, and the pitch contour -- bundled into a ``Cues``. Two encoders, one
interface:

  - ``ClassicalEncoder`` -- spectral-flux onset envelope + pyin pitch, no learning (the
    floor).
  - ``LearnedEncoder`` -- a trained backbone (TCN) + heads producing onset/voicing
    probabilities, and the pitch contour from its own ``FrequencyHead`` when the
    checkpoint carries one, falling back to pyin when it does not.

The encoder *produces* the pitch contour rather than receiving it: pitch is a cue,
and which cue-source to use (pyin, or later a learned head) is an encoder's own
choice, so it belongs in the output ``Cues``, never in as a parameter. pyin happens
to be checkpoint-independent, so an encoder memoises it (keyed on the audio) to keep
the "run pyin once, score many onset checkpoints" economy -- caching that lives
behind the encoder, not in its signature. When pitch becomes learned it is
checkpoint-*dependent* and recomputed per checkpoint, which is the point; the
interface does not change.

Swapping ``ClassicalEncoder`` for ``LearnedEncoder`` is exactly the DSP-floor-vs-learned
comparison, now a change of one composed object rather than a subclass.
"""

from abc import ABC, abstractmethod

from stemify.core.cues import Cues
from stemify.core.sources import Sources


class Encoder(ABC):
    """Turns a track's audio into ``Cues`` (onset, voicing, and pitch contour)."""

    name: str

    @abstractmethod
    def encode(self, sources: Sources) -> Cues:
        """The ``Cues`` for a track: per-frame onset (and, where the encoder has it,
        voicing) evidence derived from the audio, together with the pitch contour the
        encoder produces. Which roles of ``sources`` are read is the encoder's own
        concern; ``sources.bass`` is always available, others may be ``None``."""
