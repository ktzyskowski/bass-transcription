"""Encoders: audio -> ``Cues`` (the cue-production stage).

Only the interface is exported here. ``NeuralEncoder`` lives in ``.neural`` and must
be imported explicitly -- it pulls in torch and a trained backbone, so keeping it
out of this package's ``__init__`` means the light DSP path (``DSPEncoder``) never
transitively imports them, the same guard the decoders and old segmenters use.
"""

from stemify.encoders.base import Encoder

__all__ = ["Encoder"]
