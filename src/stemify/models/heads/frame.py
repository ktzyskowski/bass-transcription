"""Per-frame heads: a linear projection of a backbone's features to logits.

Onset, voicing and pitch differ only in the head (and its loss); the backbone is
shared. A head is a plain per-frame ``nn.Linear`` because the backbones already
return (batch, time, hidden), so all shapes are (batch, time, out_dim).
"""

import torch
from torch import nn


class FrameHead(nn.Module):
    """A per-frame linear projection to ``out_dim`` logits (no activation)."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Annotated rather than returned directly: torch types `Module.__call__` as
        # returning `Any`, so every submodule call launders its type away. Naming the
        # local puts it back at the one place that knows it.
        out: torch.Tensor = self.proj(features)   # (B, T, out_dim)
        return out


def onset_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: onset present. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)


def voicing_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: bass sounding. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)
