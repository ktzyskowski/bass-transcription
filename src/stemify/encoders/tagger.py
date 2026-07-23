"""``FrameTagger``: a backbone + named heads -> per-frame logits per head.

The trainable core a ``NeuralEncoder`` wraps. ``heads`` may hold one entry (train
onset alone) or several (joint); the backbone is shared across whatever heads are
present. The backbone is named ``backbone``, not ``encoder`` -- it is the temporal
network (TCN/GRU), distinct from the pipeline ``Encoder`` that turns audio into
``Cues``.
"""

import torch
from torch import nn

from stemify.encoders.heads import FrameHead


class FrameTagger(nn.Module):
    """Backbone + named heads -> {head_name: (B, T, out_dim) logits}.

    Any backbone exposing ``.hidden`` and returning (B, T, hidden) works -- ``TCN``
    or ``GRU``.
    """

    def __init__(self, backbone: nn.Module, heads: dict[str, FrameHead]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}
