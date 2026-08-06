"""Decoders: ``Cues`` -> notes (the decode stage).

Only the interface and the DSP-free ``GreedyDecoder`` are exported here.
``SemiCRFDecoder`` lives in ``stemify.semicrf`` -- outside ``core`` entirely, because a
trainable decoder is an *extension*, not a contract. Two things follow: the light greedy
path never transitively imports torch through this package, and ``core`` never imports
the semi-CRF at all. A caller that wants it names it.
"""

from stemify.core.decoders.base import Decoder

__all__ = ["Decoder"]
