"""Decoders: ``Cues`` -> notes (the decode stage).

Only the interface and the DSP-free ``GreedyDecoder`` are exported here.
``SemiCRFDecoder`` lives in ``.semicrf`` and must be imported explicitly -- it is a
trainable ``nn.Module`` and pulls in torch, so the light greedy path never
transitively imports it.
"""

from stemify.decoders.base import Decoder

__all__ = ["Decoder"]
