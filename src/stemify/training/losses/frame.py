"""``FrameBCELoss``: supervised per-frame BCE over fixed windows.

The encoder-head pretraining objective (no decoder in the loop): the tagger returns
per-frame logits per head, and each binary head (onset, voicing) is scored by
class-weighted BCE against its frame target, summed over heads. Onsets are a few
percent of frames, so the positive class is up-weighted. Pitch is refused until its
target encoding (plain classes vs the two-hot cents target) is settled, rather than
guessed.

This is the exact objective the old ``Trainer.build_losses``/``_batch_loss`` computed,
lifted behind the ``Loss`` interface so the same loop also drives the structured
semi-CRF objective.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from stemify.training.losses.base import Loss

BINARY_HEADS = ("onset", "voicing")


class FrameBCELoss(Loss):
    """Class-weighted per-frame BCE, summed over the binary heads being trained."""

    def __init__(
        self,
        heads: Iterable[str],
        pos_weight: dict[str, float | None] | None = None,
    ) -> None:
        self.heads = tuple(heads)
        self.pos_weight = dict(pos_weight or {})
        for h in self.heads:
            if h not in BINARY_HEADS:
                raise NotImplementedError(
                    f"No loss for head '{h}'. Binary heads {BINARY_HEADS} use weighted "
                    "BCE; pitch needs its target encoding (classes vs two-hot cents) "
                    "settled first."
                )

    def compute(
        self, model: nn.Module, batch: object, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x, y = batch  # (B, 1, n_bins, W), {head: (B, W)}
        logits = model(x.to(device))
        per_head: dict[str, torch.Tensor] = {}
        for head in self.heads:
            pred = logits[head]
            if pred.shape[-1] == 1:  # binary head: (B, W, 1) -> (B, W)
                pred = pred.squeeze(-1)
            pw = self.pos_weight.get(head)
            weight = None if pw is None else torch.tensor(pw, device=device)
            per_head[head] = F.binary_cross_entropy_with_logits(
                pred, y[head].to(device), pos_weight=weight
            )
        total = sum(per_head.values())
        # Per-head scalars plus their sum, so MLflow charts which head still learns.
        metrics = {h: float(v.item()) for h, v in per_head.items()}
        metrics["total"] = sum(metrics.values())
        return total, metrics

    def batch_size(self, batch: object) -> int:
        x, _ = batch
        return int(x.shape[0])
