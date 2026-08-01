"""The frame strategy: per-frame BCE over fixed windows, grouped by the heuristic.

The paradigm behind every result in ``docs/experiments.md`` outside the semi-CRF
section. Windows are cheap and plentiful, so this is the path with the gradient steps.

Everything one strategy needs, in one package:

  - ``recipe``  -- ``FrameRecipe``: the model, the loss, the loaders, the decoder a
    checkpoint is scored through
  - ``loss``    -- ``FrameBCELoss``, class-weighted BCE for the binary heads plus
    ``pitch_ce`` over the two-hot CQT-bin code
  - ``dataset`` -- ``FrameWindowDataset`` + ``collate``, the batch shape it consumes

These three were previously in ``recipes/``, ``losses/`` and ``datasets/``, one file
each. They are what changes together when the paradigm changes -- the batch shape, the
objective over it, and the recipe binding them -- so they now live together, and the
*contracts* they implement stay one level up in ``training.base``.
"""

from stemify.training.frame.loss import FrameBCELoss, pitch_ce
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.frame.dataset import FrameWindowDataset, collate

__all__ = [
    "FrameBCELoss",
    "FrameRecipe",
    "FrameWindowDataset",
    "collate",
    "pitch_ce",
]
