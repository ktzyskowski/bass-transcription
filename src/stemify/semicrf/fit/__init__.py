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

The *decoder* is the parent package, and the split from it is not an oversight. This
subpackage is how the thing is trained; the parent is how it decodes at inference.
Different lifetimes, different dependencies -- one ships, one does not -- so inference
can import ``stemify.semicrf`` without a dataset, a loss or a loader coming with it.
Every edge from the semi-CRF into ``training`` (``Recipe``, ``Loss``, ``TrainConfig``)
starts in here, which is what makes that claim checkable rather than aspirational.
"""

from stemify.semicrf.fit.dataset import ChunkBatch, ChunkDataset
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.semicrf.fit.recipe import SemiCRFRecipe

__all__ = [
    "ChunkBatch",
    "ChunkDataset",
    "SemiCRFLoss",
    "SemiCRFRecipe",
]
