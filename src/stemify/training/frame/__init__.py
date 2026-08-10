"""The frame strategy: per-frame BCE over fixed windows, grouped by the greedy decoder.

The paradigm behind every result in ``docs/experiments.md`` outside the semi-CRF
section. Windows are cheap and plentiful, so this is the path with the gradient steps.

Everything one strategy needs, in one package:

  - ``recipe``  -- ``FrameRecipe``: the model, the loss, the loaders, the decoder a
    checkpoint is scored through
  - ``loss``    -- ``FrameBCELoss``, class-weighted BCE for the binary heads, plus
    ``models.heads.pitch_ce`` over the two-hot CQT-bin code (that term lives with the
    head it scores, because the semi-CRF's auxiliary term needs it too)
  - ``dataset`` -- ``FrameWindowDataset`` + ``collate``, the batch shape it consumes

These three were previously in ``recipes/``, ``losses/`` and ``datasets/``, one file
each. They are what changes together when the paradigm changes -- the batch shape, the
objective over it, and the recipe binding them -- so they now live together, and the
*contracts* they implement stay one level up in ``fitting.base``.
"""

from stemify.training.frame.dataset import FrameBatch, FrameWindowDataset, collate
from stemify.training.frame.loss import FrameBCELoss
from stemify.training.frame.recipe import FrameRecipe

__all__ = [
    "FrameBCELoss",
    "FrameRecipe",
    "FrameBatch",
    "FrameWindowDataset",
    "collate",
]
