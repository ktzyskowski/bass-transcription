from abc import ABC

from stemify.decoders import Decoder
from stemify.encoders import Encoder
from stemify.notes import NoteEvent
from stemify.sources import Sources


class Segmenter(ABC):

    name: str

    def __init__(self, encoder: Encoder, decoder: Decoder):
        self.encoder = encoder
        self.decoder = decoder

    def segment(self, sources: Sources) -> list[NoteEvent]:
        cues = self.encoder.encode(sources)
        notes = self.decoder.decode(cues)
        return notes
