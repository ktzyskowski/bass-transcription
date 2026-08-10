"""Encoders: audio -> ``Frames`` (the evidence-production stage).

``Encoder`` is the contract and ``ClassicalEncoder`` (pyin + spectral flux) is the DSP
implementation. The learned one, ``LearnedEncoder``, is not here at all: it wraps a
``CueNet``, so it lives in ``stemify.models`` with the network it holds.

That split used to be a discipline inside this ``__init__`` -- export the interface,
leave the torch-bearing implementation to be imported explicitly, so the light DSP path
never transitively pulls torch. It is now a property of the layering instead: ``core``
has no torch in it anywhere, and the neural implementation is one layer up. The guard
that used to be a convention here is a test.
"""

from stemify.transcription.encoders.base import Encoder

__all__ = ["Encoder"]
