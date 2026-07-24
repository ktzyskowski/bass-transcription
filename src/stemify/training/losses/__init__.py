"""Training objectives as a strategy the ``Trainer`` is generic over.

A ``Loss`` turns a batch into a scalar to minimise, owns any learnable parameters
beyond the model (a structured decoder has its own), and names the per-component
scalars worth logging. That is the whole of what differs between training
paradigms; the loop (optimiser, epochs, early-stopping, callbacks, checkpointing)
is shared. Two paradigms live here today:

  - ``FrameBCELoss`` -- supervised per-frame BCE over fixed windows (encoder-head
    pretraining; no decoder in the loop).
  - ``SemiCRFLoss`` -- the semi-CRF's structured NLL over whole tracks, with the
    decoder *in* the loop, training encoder and decoder end to end.

They pair with a matching ``Dataset``/``collate`` (windows vs padded sequences):
the batch shape and the loss are the two things that vary, and both are ordinary
PyTorch extension points.
"""

from stemify.training.losses.base import Loss
from stemify.training.losses.frame import FrameBCELoss

__all__ = ["FrameBCELoss", "Loss"]
