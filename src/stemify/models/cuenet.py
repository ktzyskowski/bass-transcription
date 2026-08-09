"""``CueNet``: a backbone + named heads -> per-frame logits per head.

The trainable core a ``LearnedEncoder`` wraps. ``heads`` may hold one entry (train
onset alone) or several (joint); the backbone is shared across whatever heads are
present. The backbone is named ``backbone``, not ``encoder`` -- it is the temporal
network (``TCN``), distinct from the pipeline ``Encoder`` that turns audio into
``Frames``.
"""

import torch
from torch import nn

from stemify.models.heads import FrameHead
from stemify.models.networks import TCN


class CueNet(nn.Module):
    """Backbone + named heads -> {head_name: (B, T, out_dim) logits}."""

    # ``TCN``, not ``nn.Module``. This used to accept any backbone exposing ``.hidden``
    # and returning (B, T, hidden), which ``GRU`` also did -- but a *frequency* head
    # taps ``frequency_and_frame``, which only the TCN defines, so that combination
    # raised ``AttributeError`` on the first forward pass while the docstring called
    # the two interchangeable. The GRU is gone; naming the real type here is what keeps
    # a second backbone from re-introducing the same hole silently.
    backbone: TCN

    def __init__(
        self,
        backbone: TCN,
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
        ``LearnedEncoder`` holds only the tagger and still has to know how many
        channels to assemble from a track's ``Sources``.

        A direct read, not ``getattr(..., 1)``: the backbone is a ``TCN``, which always
        has this. The default was there for a backbone that might not, and it would
        have answered *1* for a two-channel model -- a plausible number, and the
        assembled input would then be the wrong width.
        """
        return self.backbone.in_channels

    def heads_and_hidden(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Every head's logits, *and* the ``(B, T, hidden)`` trunk they were read from.

        The trunk is already computed -- every head is a function of it -- and until
        this existed it was discarded at the end of the forward pass. A segmental
        decoder that pools learned features over spans wants it and not the heads: the
        heads are three scalars and a distribution, and the trunk is what they were
        summarised from, so pooling the summaries throws away the difference.

        Separate from ``forward`` rather than widening it. ``forward`` is what
        ``nn.Module.__call__`` dispatches to and what every existing caller, checkpoint
        loader and torchscript path expects to return a dict; returning a tuple there
        would make each of them unpack a second value they have no use for.
        """
        # Only pay for the frequency-resolved path when something consumes it: without
        # a frequency head the backbone's plain forward is the whole computation.
        if not self.frequency_heads:
            frame = self.backbone(x)
            return {name: head(frame) for name, head in self.heads.items()}, frame
        frequency, frame = self.backbone.frequency_and_frame(x)
        out = {name: head(frame) for name, head in self.heads.items()}
        out.update({n: h(frequency) for n, h in self.frequency_heads.items()})
        return out, frame

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads_and_hidden(x)[0]
