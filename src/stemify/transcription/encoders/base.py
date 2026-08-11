"""The ``Encoder`` interface: audio (``Sources``) -> ``Frames``.

An encoder is the *evidence-production* half of the note pipeline. It turns a track's
audio into all the per-frame evidence a decoder needs -- an onset curve, a voicing
curve, per-bin pitch logits, and the backbone representation they were read from --
bundled into a ``Frames``. One implementation today, ``LearnedEncoder`` in
``stemify.models``: a trained backbone plus heads.

The encoder *produces* the pitch contour rather than receiving it: pitch is evidence,
so it belongs in the output ``Frames``, never in as a parameter. That mattered while
there were two sources to choose between and a checkpoint-independent one worth
memoising; it still holds with one, because it is what keeps a caller from having to
know where a contour came from.

The contract stays an ABC with a single implementation because the layering is what it
buys: ``Encoder`` sits below ``models`` and knows nothing of torch, so everything from
here down -- the decoders, the metrics, the corpus -- is testable and runnable without
a network. ``tests/test_layers.py`` is what enforces that, and a second implementation
would not make it any more true.
"""

from abc import ABC, abstractmethod

from stemify.transcription.frames import Frames
from stemify.audio.sources import Sources


class Encoder(ABC):
    """Turns a track's audio into ``Frames`` (onset, voicing, and pitch contour)."""

    name: str

    @abstractmethod
    def encode(self, sources: Sources) -> Frames:
        """The ``Frames`` for a track: per-frame onset (and, where the encoder has it,
        voicing) evidence derived from the audio, together with the pitch contour the
        encoder produces. Which roles of ``sources`` are read is the encoder's own
        concern; ``sources.bass`` is always available, others may be ``None``."""
