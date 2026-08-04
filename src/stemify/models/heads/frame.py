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


def frame_bce(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float | None = None
) -> torch.Tensor:
    """BCE-with-logits over the frames whose target is finite.

    ``logits`` is ``(B, T)`` or ``(B, T, 1)``; ``target`` is ``(B, T)`` in ``{0, 1}``
    with NaN where there is nothing to learn.

    Lives beside the heads for the same reason ``pitch_ce`` does: it is the fourth
    thing about the same head (construction, forward, loss), and putting it in a
    strategy package would make it unreachable from the other strategy without one
    importing the other's internals.

    **The masking is the point**, and it is why this is not simply
    ``binary_cross_entropy_with_logits``. The frame path draws windows from inside a
    track, so every target frame is real; the semi-CRF's chunks are fixed-width and the
    last chunk of a track runs off the end, where ``target_windows`` writes NaN. Passed
    through unmasked those frames make the whole term NaN -- a silent, total failure of
    a loss that otherwise looks healthy, which is the failure mode this repository
    keeps paying for.
    """
    if logits.shape[-1] == 1:                    # (B, T, 1) -> (B, T)
        logits = logits.squeeze(-1)
    real = torch.isfinite(target)
    if not bool(real.any()):
        # Nothing to learn here, but the term has to stay in the graph or the batch
        # silently contributes no gradient rather than contributing zero.
        return logits.sum() * 0.0
    weight = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[real], target[real], pos_weight=weight
    )
