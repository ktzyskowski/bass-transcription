"""``FrameBCELoss``: supervised per-frame BCE over fixed windows.

The encoder-head pretraining objective (no decoder in the loop): the tagger returns
per-frame logits per head, and each binary head (onset, voicing) is scored by
class-weighted BCE against its frame target, summed over heads. Onsets are a few
percent of frames, so the positive class is up-weighted.

Pitch is the odd one out: its target is the two-hot CQT-bin code from
``models/heads/pitch.py``, scored by cross-entropy over the bin axis on *voiced frames
only*. Unvoiced frames are masked rather than trained toward an all-zero target --
where the bass is silent is the voicing head's question, and supervising both heads on
it would put them in competition over the shared trunk.

Sits behind the ``Loss`` interface so the same training loop also drives the structured
semi-CRF objective.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from stemify.models.heads import pitch_ce
from stemify.training.base import Loss
from stemify.training.frame.dataset import FrameBatch

BINARY_HEADS = ("onset", "voicing")
PITCH_HEAD = "pitch"
SUPPORTED_HEADS = (*BINARY_HEADS, PITCH_HEAD)


class FrameBCELoss(Loss[FrameBatch]):
    """Class-weighted per-frame BCE, summed over the binary heads being trained."""

    def __init__(
        self,
        heads: Iterable[str],
        pos_weight: dict[str, float | None] | None = None,
    ) -> None:
        self.heads = tuple(heads)
        self.pos_weight = dict(pos_weight or {})
        if not self.heads:
            # `sum()` over no heads is the int 0, which has no `backward` -- so an
            # empty list would train nothing and fail one step later, in the loop,
            # rather than here where the mistake is.
            raise ValueError("FrameBCELoss needs at least one head to score")
        for h in self.heads:
            if h not in SUPPORTED_HEADS:
                raise NotImplementedError(
                    f"No loss for head '{h}'. Supported: {SUPPORTED_HEADS} -- the "
                    "binary heads use weighted BCE, pitch uses cross-entropy over the "
                    "two-hot CQT-bin code on voiced frames."
                )

    def compute(
        self, model: nn.Module, batch: FrameBatch, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x, y = batch.x, batch.y  # (B, C, n_bins, W), {head: (B, W)}
        logits = model(x.to(device))
        per_head: dict[str, torch.Tensor] = {}
        for head in self.heads:
            pred = logits[head]
            if head == PITCH_HEAD:
                per_head[head] = pitch_ce(pred, y[head].to(device))
                continue
            if pred.shape[-1] == 1:  # binary head: (B, W, 1) -> (B, W)
                pred = pred.squeeze(-1)
            pw = self.pos_weight.get(head)
            weight = None if pw is None else torch.tensor(pw, device=device)
            per_head[head] = F.binary_cross_entropy_with_logits(
                pred, y[head].to(device), pos_weight=weight
            )
        # `torch.stack(...).sum()` rather than `sum(...)`: the builtin starts from the
        # int 0, so its result type is only a Tensor when the dict is non-empty -- which
        # the constructor now guarantees, but the reader (and the checker) cannot see
        # from here.
        total = torch.stack(list(per_head.values())).sum()
        # Per-head scalars plus their sum, so MLflow charts which head still learns.
        metrics = {h: float(v.item()) for h, v in per_head.items()}
        metrics["total"] = sum(metrics.values())
        return total, metrics

