"""The ``Decoder`` interface: ``Frames`` -> note events.

A decoder is the *decode* half of the note pipeline: it turns per-frame cues into
discrete ``NoteEvent``s, deciding where notes begin and end and assembling frames
into notes. "Decoding" is the literature's term for turning frame-level emissions
into a discrete sequence (Viterbi, CTC, greedy decoding); greedy-vs-global is the
standard dichotomy, and it is exactly the two implementations here:

  - ``GreedyDecoder`` -- greedy. Peak-pick the onset curve, threshold the voicing
    curve, split the contour at every cue (rest, pitch change, onset), then a merge
    pass rejoins same-pitch fragments a spurious cue split. The merge is a *repair*
    for the greedy pass's local mistakes.
  - ``SemiCRFDecoder`` -- a learned global decoder. One optimal segmentation of the
    whole track under learned potentials. It has no merge pass: a duration potential
    makes fragmentation globally costly, so merging is *subsumed*, not bolted on.

Decoders never invent pitch: it comes from ``frames.contour``, or from the pitch head's
own per-bin logits where a decoder scores those. They are siblings under one interface
so the same ``Frames`` can be decoded either way, and so the trainable
``SemiCRFDecoder`` can slot in where the greedy one sits.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from stemify.core.frames import Frames
from stemify.core.notes import NoteEvent


class Decoder(ABC):
    """Turns per-frame ``Frames`` into note events."""

    name: str

    @abstractmethod
    def decode(self, frames: Frames) -> list[NoteEvent]:
        """The notes implied by ``frames``."""

    def decode_batch(self, frames: Sequence[Frames]) -> list[list[NoteEvent]]:
        """Notes for several tracks at once, one list per input.

        Defaults to decoding them one at a time, which is what every decoder does
        today. It is a **default rather than an abstract method** so that callers can
        be written against the batched shape now and gain the speedup whenever a
        decoder overrides it -- no ``hasattr`` check, no ``try``/``except``, no branch
        that has to be removed later.

        Raising here instead would push the branch onto every caller, and would demand
        an implementation from ``GreedyDecoder``, where there is nothing to gain:
        it is numpy and scipy, with no per-call launch overhead to amortise. The
        decoder that *does* gain is the semi-CRF, whose DP is launch-bound and runs at
        a few hundred elements per step -- so batching it is worth roughly the batch
        factor, and the default keeps everything else working meanwhile.
        """
        return [self.decode(f) for f in frames]
