"""Truncated BPTT for the semi-CRF: the same NLL, over ordered chunks.

A *variant* of the parent strategy, not a second one. Same potentials, same DP, same
decoder; the only difference is how much of a track one gradient step sees. By the
telescoping identity the per-chunk losses sum to the whole-track NLL exactly, so the
loss value is unchanged -- what is truncated is the *gradient*, at chunk boundaries.

  - ``recipe``  -- ``SemiCRFTbpttRecipe``, which subclasses ``SemiCRFRecipe``
  - ``loss``    -- ``ChunkedSemiCRFLoss``, stateful: it carries the alpha window
  - ``dataset`` -- tracks cut into chunks, with the encoder's context margins

Why the parent still keeps a whole-track ``SemiCRFLoss`` even though this is faster and
the same objective: it is the **oracle**. ``tests/training/test_tbptt.py`` checks the
per-chunk losses telescope to one whole-track call, and that comparison is the only
thing establishing the carried-alpha bookkeeping is right. Without it, correctness
would rest on tests that compare this implementation to itself -- which is how the
right-margin carry bug survived its first round of tests.
"""

from stemify.training.semicrf.tbptt.dataset import ChunkBatch, ChunkedSequenceDataset
from stemify.training.semicrf.tbptt.loss import ChunkedSemiCRFLoss
from stemify.training.semicrf.tbptt.recipe import SemiCRFTbpttRecipe

__all__ = [
    "ChunkBatch",
    "ChunkedSemiCRFLoss",
    "ChunkedSequenceDataset",
    "SemiCRFTbpttRecipe",
]
