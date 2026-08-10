"""``Transcriber``: compose an ``Encoder`` and a ``Decoder`` into audio -> notes.

The whole pipeline in one object: ``Sources -> Encoder -> Frames -> Decoder -> notes``.
The encoder produces the per-frame evidence (the heads' curves, the backbone's own
representation, and a pitch contour); the decoder turns that into ``NoteEvent``s. Deployment holds one of
these; the note-level evaluation harness scores one of these -- so eval and deploy
transcribe through the exact same object and cannot drift.

It is deliberately not an ABC: composition is the only behaviour, and the two
swappable parts are the ``Encoder`` and ``Decoder`` it holds. The interesting
combinations are just different pairs -- ``Transcriber(ClassicalEncoder(), GreedyDecoder())``
is the DSP floor, ``Transcriber(LearnedEncoder(ckpt), SemiCRFDecoder())`` the learned
end-to-end model -- not different ``Transcriber`` subclasses.
"""

from stemify.transcription.decoders import Decoder
from stemify.transcription.encoders import Encoder
from stemify.music.notes import NoteEvent
from stemify.audio.sources import Sources


class Transcriber:
    """Encoder + Decoder as one ``Sources -> notes`` transcriber."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder

    @property
    def name(self) -> str:
        """``<encoder>+<decoder>`` -- what produced the notes, for a report label."""
        return f"{self.encoder.name}+{self.decoder.name}"

    def transcribe(self, sources: Sources) -> list[NoteEvent]:
        """The notes ``sources`` imply, via the encoder's cues and the decoder."""
        return self.decoder.decode(self.encoder.encode(sources))
