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
        heads: dict[str, FrameHead] | None = None,
        frequency_heads: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        # Two dicts because heads differ in *where they tap the network*, which is the
        # same thing as what granularity they emit. A FrameHead reads (B, T, hidden)
        # and emits one value per frame; a FrequencyHead reads the front-end's
        # (B, C, F, T), with its frequency axis intact, and emits one per bin per
        # frame. One ModuleDict cannot hold both without lying about the argument
        # shape, and a special-cased singleton would not generalise to a second
        # frequency-resolved head.
        self.heads = nn.ModuleDict(heads or {})
        self.frequency_heads = nn.ModuleDict(frequency_heads or {})

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
        # Only pay for the frequency-resolved path when something consumes it: without
        # a frequency head the backbone's plain forward is the whole computation.
        if not self.frequency_heads:
            frame = self.backbone(x)
            return {name: head(frame) for name, head in self.heads.items()}
        frequency, frame = self.backbone.frequency_and_frame(x)
        out = {name: head(frame) for name, head in self.heads.items()}
        out.update({n: h(frequency) for n, h in self.frequency_heads.items()})
        return out
