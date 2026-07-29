"""The training loop -- generic over the objective (a ``Loss``) and the batch shape.

The loop owns only what every paradigm shares: one optimiser (spanning the model
*and* any parameters the objective adds), the epoch iteration, backward/step,
early-stopping on the monitored ``"total"`` metric, callbacks and best-checkpoint
restore. What varies -- how a batch becomes a scalar loss, and what a batch even is
-- lives in the injected ``Loss`` and the loader's dataset/collate. So the same
``Trainer`` drives per-frame BCE over windows and the structured semi-CRF NLL over
padded sequences, without a branch.

Early stopping is on the monitored loss, the cheap proxy. The number that actually
matters is note-level F1 from the eval harness, but running that every epoch is
expensive; the intended use is a periodic harness eval on the side (a callback),
with the monitored loss picking the checkpoint between.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.training.callbacks import Callback, EpochLogs
from stemify.training.config import TrainConfig
from stemify.training.device import best_device
from stemify.training.losses import Loss


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


class Trainer:
    """Fit a model under a ``Loss`` with one optimiser over both their parameters."""

    def __init__(self, model: nn.Module, loss: Loss, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device or best_device())
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        # One optimiser spans the model and any parameters the objective adds (a
        # structured decoder trains jointly with the encoder); for a plain frame
        # loss, loss.parameters() is empty and this is just the model's.
        self.optimizer = torch.optim.Adam(
            chain(model.parameters(), loss.parameters()), lr=config.learning_rate
        )
        self._global_step = 0  # monotonic over train batches, for step-indexed logging

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        on_batch: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Average the objective's metrics over the epoch. Keys are the objective's
        own (per-head for BCE, ``"nll"`` for the CRF), always including ``"total"`` --
        so the caller and MLflow see the breakdown without knowing the paradigm."""
        self.model.train(train)
        totals: dict[str, float] = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                loss_t, metrics = self.loss.compute(self.model, batch, self.device)
                if train:
                    self.optimizer.zero_grad()
                    loss_t.backward()
                    self.optimizer.step()
                bs = self.loss.batch_size(batch)
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v * bs
                n += bs
                # Only train batches carry a step; val batches are averaged, not logged.
                if train and on_batch is not None:
                    self._global_step += 1
                    on_batch(self._global_step, metrics)
        return {k: v / max(n, 1) for k, v in totals.items()}

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
            # Both halves, from the *same* epoch. Restoring only the model would leave
            # the objective's parameters at their final-epoch values -- an encoder and
            # a decoder taken from different points in optimisation.
            self.load_state(best_state)
        for callback in callbacks:
            callback.on_train_end(history.best_epoch)
        return history

    def state_cpu(self) -> dict[str, dict[str, torch.Tensor]]:
        """A CPU copy of everything the optimiser is allowed to move: the model's
        weights **and** the objective's own parameters, under ``"model"``/``"loss"``.

        Both, because ``__init__`` hands the optimiser
        ``chain(model.parameters(), loss.parameters())`` -- so a snapshot of only the
        model captures part of the trained state and silently discards the rest. For
        a structured objective the discarded half is the decoder itself: restoring it
        would pair a trained encoder with freshly initialised potentials, which still
        decodes to notes and so goes unnoticed. Empty for a parameterless objective,
        which is why the frame path never noticed."""
        return {
            "model": {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            },
            "loss": self.loss.state_dict(),
        }

    def load_state(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore a ``state_cpu`` snapshot into both the model and the objective."""
        self.model.load_state_dict(state["model"])
        self.loss.load_state_dict(state["loss"])