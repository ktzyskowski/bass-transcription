"""The ``Decoder`` interface: ``Cues`` -> note events.

A decoder is the *decode* half of the note pipeline: it turns per-frame cues into
discrete ``NoteEvent``s, deciding where notes begin and end and assembling frames
into notes. "Decoding" is the literature's term for turning frame-level emissions
into a discrete sequence (Viterbi, CTC, greedy decoding); greedy-vs-global is the
standard dichotomy, and it is exactly the two implementations here:

  - ``HeuristicDecoder`` -- greedy. Peak-pick the onset curve, threshold the voicing
    curve, split the contour at every cue (rest, pitch change, onset), then a merge
    pass rejoins same-pitch fragments a spurious cue split. The merge is a *repair*
    for the greedy pass's local mistakes.
  - ``SemiCRFDecoder`` -- a learned global decoder. One optimal segmentation of the
    whole track under learned potentials. It has no merge pass: a duration potential
    makes fragmentation globally costly, so merging is *subsumed*, not bolted on.

Decoders never invent pitch -- it comes from ``cues.contour``. They are siblings
under one interface so the same ``Cues`` can be decoded either way, and so the
trainable ``SemiCRFDecoder`` can slot in where the heuristic sits.
"""

from abc import ABC, abstractmethod

from stemify.cues import Cues
from stemify.notes import NoteEvent


class Decoder(ABC):
    """Turns per-frame ``Cues`` into note events."""

    name: str

    @abstractmethod
    def decode(self, cues: Cues) -> list[NoteEvent]:
        """The notes implied by ``cues``."""
