"""The training loop -- head-agnostic, driven by a loss per head.

The model returns a dict of per-frame logits, one entry per head; the trainer
sums a per-head loss over whichever heads are present, so training one head
(onset first) or several (joint) is the same code. Onset and voicing are binary
per-frame, so their loss is BCE-with-logits, class-weighted because onsets are a
few percent of frames. Pitch is deferred: its target encoding (plain classes vs
the two-hot cents target) is still open, so ``build_losses`` refuses it rather
than guess.

Early stopping is on validation loss, the cheap proxy. The number that actually
matters is note-level F1 from the eval harness, but running that every epoch is
expensive; the intended use is a periodic harness eval on the side, with val loss
picking the checkpoint between.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.training.callbacks import Callback, EpochLogs
from stemify.training.config import TrainConfig
from stemify.training.device import best_device
from stemify.training.model import FrameTagger

BINARY_HEADS = ("onset", "voicing")


def build_losses(
    heads: Iterable[str],
    pos_weight: dict[str, float | None],
    device: torch.device,
) -> dict[str, nn.Module]:
    """A loss module per head. Binary heads get weighted BCE; pitch is refused
    until its encoding is decided."""
    losses: dict[str, nn.Module] = {}
    for h in heads:
        if h in BINARY_HEADS:
            pw = pos_weight.get(h)
            weight = None if pw is None else torch.tensor(pw, device=device)
            losses[h] = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            raise NotImplementedError(
                f"No loss for head '{h}'. Binary heads {BINARY_HEADS} use weighted "
                "BCE; pitch needs its target encoding (classes vs two-hot cents) "
                "settled first."
            )
    return losses


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


class Trainer:
    """Fit a ``FrameTagger`` with one optimiser and a per-head loss sum."""

    def __init__(self, model: FrameTagger, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device or best_device())
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.losses = build_losses(config.heads, config.pos_weight, self.device)
        self._global_step = 0  # monotonic over train batches, for step-indexed logging

    def _batch_loss(
        self, x: torch.Tensor, y: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Per-head loss tensors. The caller sums them for backward and syncs each
        for logging, so onset and voicing are tracked separately, not only summed."""
        logits = self.model(x.to(self.device))
        per_head: dict[str, torch.Tensor] = {}
        for head, loss_fn in self.losses.items():
            pred = logits[head]
            if pred.shape[-1] == 1:  # binary head: (B, T, 1) -> (B, T)
                pred = pred.squeeze(-1)
            per_head[head] = loss_fn(pred, y[head].to(self.device))
        return per_head

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        on_batch: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Average loss over the epoch as ``{"total": ..., <head>: ...}`` -- per-head
        so the caller (and MLflow) can see which head is still learning."""
        self.model.train(train)
        totals: dict[str, float] = {}
        n = 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                per_head = self._batch_loss(x, y)
                loss = sum(per_head.values())  # tensor sum, for backward
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                # One .item() per head is the only device sync (total is their sum,
                # needing no extra one); was a single sync for onset alone.
                batch = {h: v.detach().item() for h, v in per_head.items()}
                batch["total"] = sum(batch.values())
                bs = x.shape[0]
                for h, v in batch.items():
                    totals[h] = totals.get(h, 0.0) + v * bs
                n += bs
                # Only train batches carry a step; val batches are averaged, not logged.
                if train and on_batch is not None:
                    self._global_step += 1
                    on_batch(self._global_step, batch)
        return {h: v / max(n, 1) for h, v in totals.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: Iterable[Callback] = (),
    ) -> History:
        callbacks = list(callbacks)
        history = History()
        best_val = float("inf")
        best_state = None
        stale = 0

        def emit_batch(step: int, losses: dict[str, float]) -> None:
            for callback in callbacks:
                callback.on_batch_end(step, losses)

        for epoch in range(self.config.max_epochs):
            train_losses = self._run_epoch(train_loader, train=True, on_batch=emit_batch)
            val_losses = (
                self._run_epoch(val_loader, train=False)
                if val_loader is not None
                else train_losses
            )
            train_loss, monitored = train_losses["total"], val_losses["total"]
            history.train_loss.append(train_loss)
            history.val_loss.append(monitored)

            if monitored < best_val - 1e-5:
                best_val, best_state, history.best_epoch, stale = (
                    monitored, self.state_cpu(), epoch, 0
                )
            else:
                stale += 1

            logs = EpochLogs(
                epoch=epoch, train_loss=train_loss, val_loss=monitored,
                train_losses=train_losses, val_losses=val_losses,
                best_epoch=history.best_epoch, snapshot=self.state_cpu,
            )
            for callback in callbacks:
                callback.on_epoch_end(logs)

            if stale >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        for callback in callbacks:
            callback.on_train_end(history.best_epoch)
        return history

    def state_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}