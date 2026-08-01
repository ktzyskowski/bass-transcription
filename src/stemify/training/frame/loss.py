"""``FrameBCELoss``: supervised per-frame BCE over fixed windows.

The encoder-head pretraining objective (no decoder in the loop): the tagger returns
per-frame logits per head, and each binary head (onset, voicing) is scored by
class-weighted BCE against its frame target, summed over heads. Onsets are a few
percent of frames, so the positive class is up-weighted.

Pitch was refused here until its target encoding was settled; it is now the two-hot
CQT-bin code in ``models/heads/pitch.py``, scored by cross-entropy over the bin axis on
*voiced frames only*. Unvoiced frames are masked rather than trained toward an all-zero
target: where the bass is silent is the voicing head's question, and supervising both
heads on it would put them in competition over the shared trunk.

This is the exact objective the old ``Trainer.build_losses``/``_batch_loss`` computed,
lifted behind the ``Loss`` interface so the same loop also drives the structured
semi-CRF objective.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from stemify.models.heads import two_hot
from stemify.training.base import Loss

BINARY_HEADS = ("onset", "voicing")
PITCH_HEAD = "pitch"
SUPPORTED_HEADS = (*BINARY_HEADS, PITCH_HEAD)


def pitch_ce(logits: torch.Tensor, midi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the CQT bin axis against the two-hot code, voiced frames only.

    ``logits`` is ``(B, T, n_bins)``; ``midi`` is ``(B, T)`` continuous MIDI with NaN
    where the bass is silent.

    **Why cross-entropy and not per-bin BCE.** The two-hot target is a proper
    distribution -- its rows sum to 1 -- so the axis is a choice among bins, not 216
    independent yes/no questions. Treating it as the latter is not merely a different
    parameterisation, it puts the term on the wrong *scale*: 216 independent sigmoids
    start near ``216 * ln 2 ~ 150``, against ``ln 216 ~ 5.4`` for a choice among 216
    options. The first version of this summed BCE and measured 108.4 against onset
    0.249 and voicing 0.267 -- a ~400x imbalance on a shared trunk, which no loss
    weight should have to correct. This is the same reason DreamerV3 scores its
    two-hot critic and reward heads with categorical cross-entropy: get the functional
    form right and fixed coefficients work across scales.

    Unvoiced frames are dropped rather than trained toward some "no pitch" target.
    Where the bass sounds is the voicing head's question, and supervising this head on
    it too would have the two compete over a shared trunk to answer the same thing.
    """
    voiced = ~torch.isnan(midi)
    if not bool(voiced.any()):
        # Nothing to learn from this batch, but the term must stay in the graph or the
        # optimiser sees a different set of parameters depending on the batch drawn.
        return logits.sum() * 0.0
    # Soft-target cross-entropy: -sum_b target[b] * log q[b]. F.cross_entropy accepts
    # probability targets too, but only with the class axis at dim 1 -- spelling it out
    # keeps the bin axis last, matching two_hot and the head's own output.
    per_frame = -(two_hot(midi) * F.log_softmax(logits, dim=-1)).sum(-1)
    return (per_frame * voiced).sum() / voiced.sum()


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
            if h not in SUPPORTED_HEADS:
                raise NotImplementedError(
                    f"No loss for head '{h}'. Supported: {SUPPORTED_HEADS} -- the "
                    "binary heads use weighted BCE, pitch uses cross-entropy over the "
                    "two-hot CQT-bin code on voiced frames."
                )

    def compute(
        self, model: nn.Module, batch: object, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x, y = batch  # (B, 1, n_bins, W), {head: (B, W)}
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
        total = sum(per_head.values())
        # Per-head scalars plus their sum, so MLflow charts which head still learns.
        metrics = {h: float(v.item()) for h, v in per_head.items()}
        metrics["total"] = sum(metrics.values())
        return total, metrics

    def batch_size(self, batch: object) -> int:
        x, _ = batch
        return int(x.shape[0])
