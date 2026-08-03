"""The semi-CRF strategy: structured NLL over ordered chunks, decoder in the loop.

Parked (see ``docs/ablation-plan.md``), and the parked verdict is narrower than it
reads: what lost was the *weakest form* of the idea -- a handful of linear scalars
weighting hand-crafted span features -- not a neural segment scorer. When it is
revisited, that potential network belongs in ``models``; this package is the training
side.

  - ``recipe``  -- ``SemiCRFRecipe``
  - ``loss``    -- ``SemiCRFLoss``, the segmental forward NLL, carrying alpha between
    a track's chunks. ``detach_carry=False`` recovers untruncated whole-track backprop.
  - ``dataset`` -- tracks cut into fixed chunks, with the encoder's context margins
  - ``tracks``  -- track-input resolution

This used to be a package with a ``tbptt`` subpackage under it, each with its own
dataset, batch type, loss and recipe, computing the same objective two ways. They
differed on one axis -- whether the alpha carry is detached between chunks -- so that
is a flag on the loss now, and there is one of each again.

The *decoder* stays in ``decoders.semicrf``, and that is not an oversight. This package
is how the thing is trained; that one is how it decodes at inference. Different
lifetimes, different dependencies -- one ships, one does not.
"""

from stemify.training.semicrf.dataset import ChunkBatch, ChunkDataset
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.semicrf.recipe import SemiCRFRecipe

__all__ = [
    "ChunkBatch",
    "ChunkDataset",
    "SemiCRFLoss",
    "SemiCRFRecipe",
]
