"""The recipe table: every training paradigm a run can name, by tag.

A ``Recipe`` binds the four choices that have to agree -- batch shape, loss, the
decoder a checkpoint is scored through, and what that checkpoint must contain. The
contracts live in ``training.base``; each paradigm lives in its own *package*
(``training.frame``, ``training.semicrf``); this assembles them so ``--objective`` is
the only thing a caller spells.

Kept apart from ``base`` on purpose: this module imports the strategy packages and they
import ``base``, so folding the table in beside the ABCs would close the cycle.
"""

from stemify.training.base import HEADS, Recipe
from stemify.training.frame import FrameRecipe
from stemify.training.semicrf import SemiCRFRecipe

RECIPES: dict[str, type[Recipe]] = {r.name: r for r in (FrameRecipe, SemiCRFRecipe)}

__all__ = ["HEADS", "RECIPES", "FrameRecipe", "Recipe", "SemiCRFRecipe"]
