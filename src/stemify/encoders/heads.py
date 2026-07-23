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
        return self.proj(features)  # (B, T, out_dim)


def onset_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: onset present. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)


def voicing_head(hidden: int = 128) -> FrameHead:
    """One logit per frame: bass sounding. BCE-with-logits at train time."""
    return FrameHead(hidden, 1)


def pitch_head(hidden: int = 128, n_classes: int = 45) -> FrameHead:
    """One logit per pitch class per frame (default 45 = MIDI 23-67, the 5-string
    range the confusion matrix already uses). Cross-entropy, or swap for the
    two-hot cents target later."""
    return FrameHead(hidden, n_classes)
