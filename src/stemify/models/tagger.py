"""``FrameTagger``: a backbone + named heads -> per-frame logits per head.

The trainable core a ``NeuralEncoder`` wraps. ``heads`` may hold one entry (train
onset alone) or several (joint); the backbone is shared across whatever heads are
present. The backbone is named ``backbone``, not ``encoder`` -- it is the temporal
network (TCN/GRU), distinct from the pipeline ``Encoder`` that turns audio into
``Cues``.
"""

import torch
from torch import nn

from stemify.models.heads import FrameHead


class FrameTagger(nn.Module):
    """Backbone + named heads -> {head_name: (B, T, out_dim) logits}.

    Any backbone exposing ``.hidden`` and returning (B, T, hidden) works -- ``TCN``
    or ``GRU``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: dict[str, FrameHead],
        pitch: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        # Held apart from ``heads`` because it consumes a different tensor: the per-
        # frame heads read (B, T, hidden), the pitch branch reads the front-end's
        # (B, C, F, T) with its frequency axis intact. Same forward, same optimiser,
        # different input -- so a ModuleDict of things taking one argument shape
        # cannot hold both without lying about it.
        self.pitch = pitch

    @property
    def in_channels(self) -> int:
        """How many stacked renderings the input carries (see ``_ConvFrontend``).

        Read off the backbone rather than stored, so it cannot drift from the conv
        that actually has to accept the tensor. Surfaced here because a
        ``NeuralEncoder`` holds only the tagger and still has to know how many
        channels to assemble from a track's ``Sources``.
        """
        return getattr(self.backbone, "in_channels", 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.pitch is None:
            features = self.backbone(x)
            return {name: head(features) for name, head in self.heads.items()}
        spectral, features = self.backbone.spectral_and_temporal(x)
        out = {name: head(features) for name, head in self.heads.items()}
        out["pitch"] = self.pitch(spectral)
        return out
