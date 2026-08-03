"""``Segmenter``: compose an ``Encoder`` and a ``Decoder`` into audio -> notes.

The whole pipeline in one object: ``Sources -> Encoder -> Cues -> Decoder -> notes``.
The encoder produces the per-frame cues (onset/voicing evidence + the pitch contour);
the decoder turns those cues into discrete ``NoteEvent``s. Deployment holds one of
these; the note-level evaluation harness scores one of these -- so eval and deploy
transcribe through the exact same object and cannot drift.

It is deliberately not an ABC: composition is the only behaviour, and the two
swappable parts are the ``Encoder`` and ``Decoder`` it holds. The interesting
combinations are just different pairs -- ``Segmenter(ClassicalEncoder(), HeuristicDecoder())``
is the DSP floor, ``Segmenter(LearnedEncoder(ckpt), SemiCRFDecoder())`` the learned
end-to-end model -- not different ``Segmenter`` subclasses.
"""

from stemify.decoders import Decoder
from stemify.encoders import Encoder
from stemify.notes import NoteEvent
from stemify.sources import Sources


class Segmenter:
    """Encoder + Decoder as one ``Sources -> notes`` transcriber."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder

    @property
    def name(self) -> str:
        """``<encoder>+<decoder>`` -- what produced the notes, for a report label."""
        return f"{self.encoder.name}+{self.decoder.name}"

    def segment(self, sources: Sources) -> list[NoteEvent]:
        """The notes ``sources`` imply, via the encoder's cues and the decoder."""
        return self.decoder.decode(self.encoder.encode(sources))
