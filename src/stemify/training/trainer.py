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

Divergence is treated as a bug, not a bad epoch. Once a loss is NaN every downstream
number stays NaN and nothing here notices on its own: ``nan < best_val`` is False, so
early stopping never fires and never checkpoints, the epoch log prints ``nan`` beside
its other ``nan``s, and the loop runs to ``max_epochs`` and saves whatever it holds
while reporting success. So the loop clips gradients (``TrainConfig.max_grad_norm``,
over everything the optimiser moves) and refuses to continue past a non-finite loss or
gradient -- the first raising where it happens, naming the step and the term.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Generic

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.device import best_device
from stemify.training.base import GRAD_NORM, B, Loss
from stemify.training.callbacks import Callback, EpochLogs
from stemify.training.config import TrainConfig


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


def _nonfinite(values: Mapping[str, float]) -> list[str]:
    """The keys whose value is NaN or +-inf, in the order given."""
    return [k for k, v in values.items() if not math.isfinite(v)]


class Trainer(Generic[B]):
    """Fit a model under a ``Loss`` with one optimiser over both their parameters.

    Generic in the batch type for the same reason ``Loss`` is: a loss and the loader
    feeding it are a matched pair, and ``Recipe`` builds them together precisely so
    they cannot be mixed. Spelling the pairing here means a ``Loss[ChunkBatch]`` handed
    a frame loader is rejected where the two meet, rather than inside ``compute``.
    """

    def __init__(self, model: nn.Module, loss: Loss[B], config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device(config.device or best_device())
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        # One optimiser spans the model and any parameters the objective adds (a
        # structured decoder trains jointly with the encoder); for a plain frame
        # loss, loss.parameters() is empty and this is just the model's.
        # Materialised rather than left as the chain iterator, because clipping needs
        # the same set every step and an iterator is empty after the optimiser drains it.
        self._params = list(chain(self.model.parameters(), self.loss.parameters()))
        self.optimizer = torch.optim.Adam(self._params, lr=config.learning_rate)
        self._global_step = 0  # monotonic over train batches, for step-indexed logging

    def _clip(self) -> float:
        """Clip the accumulated gradient in place; return its norm *before* clipping.

        The pre-clip norm is the number worth logging: post-clip it is either itself or
        exactly ``max_grad_norm``, which says nothing. Returned rather than logged here
        so it rides out on the batch metrics and reaches MLflow through the path the
        loss terms already use.
        """
        limit = self.config.max_grad_norm
        # No clip still measures. The metric is the whole reason a threshold can be
        # chosen from data, so disabling the clip must not also blind the run.
        norm = torch.nn.utils.clip_grad_norm_(
            self._params, float("inf") if limit is None else limit
        )
        return float(norm.item())

    def _require_finite(
        self, values: Mapping[str, float], train: bool, kind: str
    ) -> None:
        """Refuse to continue past a NaN or infinite ``kind``, naming the step and which
        terms went bad.

        Raising, rather than skipping the batch: by the time a loss is non-finite the
        weights that produced it almost always already are, so the next step is not
        recoverable -- and a silently skipped step is the failure mode this loop exists
        to make loud. The step number and the surviving finite terms are the diagnosis:
        every term going at once implicates the shared trunk, one term alone implicates
        that head.
        """
        bad = _nonfinite(values)
        if not bad:
            return
        phase = "training" if train else "validation"
        raise ValueError(
            f"non-finite {kind} during {phase} at global step {self._global_step}: "
            f"{', '.join(f'{k}={values[k]}' for k in bad)}"
            + (f" (finite: {', '.join(k for k in values if k not in bad)})"
               if len(bad) < len(values) else "")
            + ". Training has diverged and every number after this point is NaN, so "
            "the loop stops here rather than running on and saving a NaN checkpoint. "
            "Lower --learning-rate, or lower --max-grad-norm (see the logged "
            f"{GRAD_NORM!r} metric for what the norm was doing before this step)."
        )

    def _run_epoch(
        self,
        loader: DataLoader[B],
        train: bool,
        on_batch: Callable[[int, dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Average the objective's metrics over the epoch. Keys are the objective's
        own (per-head for BCE, ``"nll"`` for the CRF), always including ``"total"`` --
        so the caller and MLflow see the breakdown without knowing the paradigm.

        Train epochs carry one key the objective did not produce, ``GRAD_NORM``; its
        epoch mean answers whether the clip is binding at all, which is what makes
        ``max_grad_norm`` a number chosen from the run rather than a guess."""
        self.model.train(train)
        totals: dict[str, float] = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                # Counted for every train batch, not only when someone is listening:
                # the step is where the loop is, and the divergence guard names it.
                if train:
                    self._global_step += 1
                loss_t, metrics = self.loss.compute(self.model, batch, self.device)
                # Before backward, so a NaN is reported at the step that produced it
                # rather than one step later as a NaN gradient. Free: every `Loss`
                # already materialises these to Python floats, so the sync is paid.
                self._require_finite(metrics, train, "loss")
                if train:
                    self.optimizer.zero_grad()
                    # torch ships `Tensor.backward` unannotated, so --strict calls this
                    # an untyped call in typed context. Nothing to fix on our side.
                    loss_t.backward()  # type: ignore[no-untyped-call]
                    norm = self._clip()
                    metrics = {**metrics, GRAD_NORM: norm}
                    # A finite loss can still backward into a non-finite gradient, and
                    # clipping cannot rescue one -- scaling inf or NaN leaves it. Check
                    # between clip and step, the last point where the weights are good.
                    self._require_finite({GRAD_NORM: norm}, train, "gradient")
                    self.optimizer.step()
                bs = len(batch)
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v * bs
                n += bs
                # Only train batches carry a step; val batches are averaged, not logged.
                if train and on_batch is not None:
                    on_batch(self._global_step, metrics)
        if n == 0:
            # A loader that yields nothing returns no metrics, so the caller's
            # `losses["total"]` raises a bare KeyError with no hint that the *loader*
            # was empty. Every reachable cause is a misconfiguration -- a split with no
            # tracks, a limit below the batch size, a filter that excluded everything --
            # so it is worth failing here, where the emptiness is the observation.
            raise ValueError(
                f"the {'train' if train else 'validation'} loader yielded no batches, "
                "so this epoch has nothing to average. Usual causes: the split is "
                "empty, or a --limit/--val-limit is smaller than the batch size."
            )
        return {k: v / n for k, v in totals.items()}

    def fit(
        self,
        train_loader: DataLoader[B],
        val_loader: DataLoader[B] | None = None,
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