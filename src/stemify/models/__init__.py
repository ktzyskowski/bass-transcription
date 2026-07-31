"""The trainable machinery: backbones, heads, and the tagger that composes them.

Split out of ``encoders/`` because that package had come to hold two unrelated things.
An ``Encoder`` is a *pipeline stage* -- audio in, ``Cues`` out -- and there are two of
them, ``DSPEncoder`` and ``NeuralEncoder``. Everything here is the neural network one
of those two happens to wrap: ``DSPEncoder`` touches none of it, and ``training/`` was
reaching into ``encoders.networks`` and ``encoders.heads`` for things that are not
encoders at all, which is the smell that gave the mixing away.

The dependency direction is now one-way and says what it means:

    models/      knows nothing about encoders, decoders or training
    encoders/    NeuralEncoder wraps a FrameTagger to produce Cues
    training/    trains a FrameTagger; imports models directly, not through encoders

``FrameTagger`` is the composition point -- a backbone plus named heads -- and it is
what a checkpoint stores. Everything is re-exported here, since a caller wanting a
network invariably wants a head and a tagger too.

Unlike ``encoders``, this package imports torch eagerly: there is nothing in it that
does not need torch, so there is no light path to protect. That guard belongs one
level up, in ``encoders/__init__.py``, which is exactly why it still lives there.
"""

from stemify.models.heads import (
    FrameHead,
    FrequencyHead,
    decode_bins,
    decode_contour,
    onset_head,
    pitch_head,
    two_hot,
    voicing_head,
)
from stemify.models.networks import GRU, TCN, tcn_receptive_field
from stemify.models.tagger import FrameTagger

__all__ = [
    "GRU",
    "TCN",
    "FrameHead",
    "FrameTagger",
    "FrequencyHead",
    "decode_bins",
    "decode_contour",
    "onset_head",
    "pitch_head",
    "tcn_receptive_field",
    "two_hot",
    "voicing_head",
]
