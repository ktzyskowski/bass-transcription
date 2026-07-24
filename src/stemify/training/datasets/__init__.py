"""Training datasets, one module per batch shape.

The two training paradigms consume differently shaped batches, so each gets its own
``torch.utils.data.Dataset`` + ``collate``: ``frame`` serves fixed windows for the
per-frame ``FrameBCELoss``; ``sequence`` serves whole padded tracks for the
structured ``SemiCRFLoss``. The batch shape and the loss are the two things that vary
between paradigms; the ``Trainer`` is shared. Each module keeps its own ``collate``
(imported from the specific module, since there are two)."""

from stemify.training.datasets.frame import FrameWindowDataset
from stemify.training.datasets.sequence import (
    SequenceBatch,
    SequenceDataset,
    SequenceExample,
)

__all__ = [
    "FrameWindowDataset",
    "SequenceBatch",
    "SequenceDataset",
    "SequenceExample",
]
