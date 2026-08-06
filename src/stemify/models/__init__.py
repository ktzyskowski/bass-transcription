"""The neural layer: backbones, heads, and the two pipeline stages that wrap one.

Sits between ``core`` and ``training``, and the boundary below it is the sharp one:
**``core`` contains no torch at all, and everything here needs it.** That is not a
convention -- ``tests/test_layers.py`` asserts that importing ``stemify.core`` leaves
``torch`` out of ``sys.modules``, so the day something reaches downward it fails loudly
rather than quietly re-coupling the two.

Why this is its own layer rather than part of ``training``: a trained network at
inference is a *pure function*. Both wrappers here run under ``@torch.no_grad()`` on a
model in ``.eval()``, and ``stemify-transcribe`` needs them. Putting them in
``training`` would mean transcribing a wav imports the trainer. Autograd belongs to
``Recipe``/``Loss``/``Trainer``; an ``nn.Module`` is not autograd. The seam is the
*model* (here) against the *fitting procedure* (there).

Why it is not part of ``core`` either: a contract should be satisfiable without the
thing satisfying it. ``Encoder`` and ``PitchEstimator`` are core ABCs precisely so that
``ClassicalEncoder``/``PyinPitch`` (DSP, no torch) and ``LearnedEncoder``/``NeuralPitch``
(a ``CueNet``) can both wear them without either knowing about the other.

    core/       Encoder, PitchEstimator, Decoder, Cues, Contour -- the contracts
    models/     TCN -> CueNet -> heads, plus LearnedEncoder and NeuralPitch
    training/   fits a CueNet; imports this directly, never through an encoder

``CueNet`` is the composition point -- a backbone plus named heads -- and it is what a
checkpoint stores. Everything is re-exported, since a caller wanting a network
invariably wants a head and a wrapper too.
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
from stemify.models.networks import TCN, tcn_receptive_field
from stemify.models.cuenet import CueNet
from stemify.models.learned_encoder import LearnedEncoder
from stemify.models.neural_pitch import NeuralPitch

__all__ = [
    "TCN",
    "FrameHead",
    "CueNet",
    "FrequencyHead",
    "LearnedEncoder",
    "NeuralPitch",
    "decode_bins",
    "decode_contour",
    "onset_head",
    "pitch_head",
    "tcn_receptive_field",
    "two_hot",
    "voicing_head",
]
