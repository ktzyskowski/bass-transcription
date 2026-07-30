"""The recipe table: every training paradigm a run can name, by tag.

A ``Recipe`` binds the four choices that have to agree -- batch shape, loss, the
decoder a checkpoint is scored through, and what that checkpoint must contain. The
seam and the shared parts live in ``base``; each paradigm lives in its own module; this
assembles them so ``--objective`` is the only thing a caller spells.
"""

from stemify.training.recipes.base import HEADS, Recipe
from stemify.training.recipes.frame import FrameRecipe
from stemify.training.recipes.semicrf import SemiCRFRecipe, SemiCRFTbpttRecipe

RECIPES: dict[str, type[Recipe]] = {
    r.name: r for r in (FrameRecipe, SemiCRFRecipe, SemiCRFTbpttRecipe)
}

__all__ = [
    "HEADS", "RECIPES", "FrameRecipe", "Recipe", "SemiCRFRecipe", "SemiCRFTbpttRecipe",
]
