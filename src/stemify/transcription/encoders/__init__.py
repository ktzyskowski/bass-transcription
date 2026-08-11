"""Encoders: audio -> ``Frames`` (the evidence-production stage).

``Encoder`` is the contract. Its one implementation, ``LearnedEncoder``, is not here:
it wraps a ``FrameNet``, so it lives in ``stemify.models`` with the network it holds,
which is what keeps this layer free of torch.

``ClassicalEncoder`` -- spectral flux plus a pyin contour, the DSP floor -- used to sit
beside the contract as the torch-free implementation that proved the split was real.
It went when pyin did: its pitch source no longer exists, and the floor's numbers are
recorded in docs/experiments.md rather than re-derived. The guard that the lower layers
stay torch-free is now a test (``tests/test_layers.py``) rather than a resident example.
"""

from stemify.transcription.encoders.base import Encoder

__all__ = ["Encoder"]
