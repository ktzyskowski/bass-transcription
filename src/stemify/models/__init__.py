"""The neural layer: backbones, heads, and the two pipeline stages that wrap one.

Sits above the pillars and below ``fitting``, and the boundary beneath it is the sharp
one: **no pillar contains torch at all, and everything here needs it.** That is not a
convention -- ``tests/test_layers.py`` walks each pillar in a subprocess and asserts that
importing it leaves ``torch`` out of ``sys.modules``, so the day something reaches
downward it fails loudly rather than quietly re-coupling the two.

In particular this implements ``transcription``'s contracts (``LearnedEncoder`` is an
``Encoder``) and is depended *on* by ``fitting``,
whose ``Recipe.build_model()`` returns a ``FrameNet``. That is the whole reason those
two contract sets are different layers.

Why this is its own layer rather than part of ``training``: a trained network at
inference is a *pure function*. Both wrappers here run under ``@torch.no_grad()`` on a
model in ``.eval()``, and ``stemify-transcribe`` needs them. Putting them in
``training`` would mean transcribing a wav imports the trainer. Autograd belongs to
``Recipe``/``Loss``/``Trainer``; an ``nn.Module`` is not autograd. The seam is the
*model* (here) against the *fitting procedure* (there).

Why it is not part of ``core`` either: a contract should be satisfiable without the
thing satisfying it. ``Encoder`` is a contract in the layer below precisely so that
``LearnedEncoder``
(a ``FrameNet``) can both wear them without either knowing about the other.

    core/       Encoder, Decoder, Frames, Contour -- the contracts
    models/     TCN -> FrameNet -> heads, plus LearnedEncoder
    training/   fits a FrameNet; imports this directly, never through an encoder

``FrameNet`` is the composition point -- a backbone plus named heads -- and it is what a
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
from stemify.models.framenet import FrameNet
from stemify.models.learned_encoder import LearnedEncoder

__all__ = [
    "TCN",
    "FrameHead",
    "FrameNet",
    "FrequencyHead",
    "LearnedEncoder",
    "decode_bins",
    "decode_contour",
    "onset_head",
    "pitch_head",
    "tcn_receptive_field",
    "two_hot",
    "voicing_head",
]
