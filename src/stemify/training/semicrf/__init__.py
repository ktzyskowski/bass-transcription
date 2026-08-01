"""The semi-CRF strategy: structured NLL over whole tracks, decoder in the loop.

Parked (see ``docs/ablation-plan.md``), and the parked verdict is narrower than it
reads: what lost was the *weakest form* of the idea -- 20 linear scalars weighting three
hand-crafted features, one of them built on pyin's lagged pitch transitions -- not a
neural segment scorer. When it is revisited, that potential network belongs in
``models``; this package is the training side.

  - ``recipe``  -- ``SemiCRFRecipe``
  - ``loss``    -- ``SemiCRFLoss``, the segmental forward NLL over a whole track
  - ``dataset`` -- whole padded tracks
  - ``tracks``  -- track-input resolution, shared with the variant below

``tbptt`` is a **subpackage**, not a sibling: truncated BPTT is the same objective --
same potentials, same DP, same decoder -- differing only in how much of a track one
gradient step sees, and its recipe subclasses ``SemiCRFRecipe``. Nesting says "variant
of"; a sibling package would have said "second strategy" and pointed the imports the
wrong way.

It is deliberately *not* re-exported here. ``recipes`` imports it explicitly, which
keeps this ``__init__`` from importing a subpackage that imports back into it.

The *decoder* stays in ``decoders.semicrf``, and that is not an oversight. This package
is how the thing is trained; that one is how it decodes at inference. Different
lifetimes, different dependencies -- one ships, one does not.
"""

from stemify.training.semicrf.dataset import (
    SequenceBatch,
    SequenceDataset,
    SequenceExample,
)
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.semicrf.recipe import SemiCRFRecipe

__all__ = [
    "SemiCRFLoss",
    "SemiCRFRecipe",
    "SequenceBatch",
    "SequenceDataset",
    "SequenceExample",
]
