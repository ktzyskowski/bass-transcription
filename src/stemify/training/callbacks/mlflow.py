"""``MLflowCallback``: log train/val loss and async note-level metrics to MLflow.

Uses ``MlflowClient`` (not the global fluent API), so the async validator's worker
thread and the main loop can log to the same run at once without racing on a shared
active run. ``mlflow`` is imported lazily inside ``__init__`` -- tracking is optional
and the trainer never pulls it in. (``import mlflow`` here is absolute, resolving to
the installed library, not this module.)
"""

import logging
import os
from collections.abc import Mapping
from typing import Any

from stemify.training.base import GRAD_NORM
from stemify.training.callbacks.base import Callback, EpochLogs, Metrics
from stemify.training.ema import EMA

logger = logging.getLogger(__name__)


def _metric_key(key: str, phase: str) -> str:
    """Slash-namespace one batch/epoch metric for MLflow's chart grouping.

    Loss terms go under ``loss/`` so the heads share a section; the loop's gradient
    norm gets its own, because filing a norm under ``loss`` puts a number that is not
    a loss on the same axis as the ones that are.
    """
    return f"{key}/{phase}" if key == GRAD_NORM else f"loss/{key}/{phase}"


class MLflowCallback(Callback):
    """Log train/val loss (and, via ``sink``, async note-level metrics) to MLflow."""

    def __init__(
        self,
        run_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tracking_uri: str | None = None,
        experiment: str | None = None,
        batch_log_every: int = 100,
        ema_beta: float = 0.98,
    ) -> None:
        import mlflow  # lazy: tracking is optional, the trainer never imports it
        from mlflow.tracking import MlflowClient

        # Per-epoch loss (one point/epoch) is too coarse to see divergence inside a
        # ~50k-step epoch, but raw per-batch BCE is noisy; log a smoothed loss every
        # ``batch_log_every`` steps instead (0 disables). One EMA per head (created on
        # first sight), so onset and voicing get their own step curve, not just total.
        self._batch_log_every = batch_log_every
        self._ema_beta = ema_beta
        self._emas: dict[str, EMA] = {}

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        # Get-or-create rather than assume experiment "0" exists: on a fresh (or
        # pre-created empty) store the Default experiment has not been
        # initialised yet, and create_run would fail looking it up.
        name = experiment or "Default"
        found = self._client.get_experiment_by_name(name)
        experiment_id = found.experiment_id if found else self._client.create_experiment(name)
        self._run_id = self._client.create_run(experiment_id, run_name=run_name).info.run_id
        for key, value in (params or {}).items():
            self._client.log_param(self._run_id, key, value)

    def sink(self, metrics: Metrics, step: int) -> None:
        """Thread-safe metric logging -- the entry point for the async validator."""
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, float(value), step=step)

    def on_batch_end(self, step: int, losses: Mapping[str, float]) -> None:
        for head, value in losses.items():
            self._emas.setdefault(head, EMA(self._ema_beta)).update(value)
        if self._batch_log_every and step % self._batch_log_every == 0:
            # Every EMA seen so far, not just the keys on *this* batch. Under gradient
            # accumulation `grad_norm` exists only on optimiser steps, so keying off
            # the current batch would log it solely when the two periods coincide --
            # every lcm(accumulate_steps, batch_log_every) steps, which for 53 and 100
            # is three points an epoch. The EMA already holds the last value; that is
            # what it is for.
            self.sink(
                {
                    _metric_key(h, "step"): e.value
                    for h, e in self._emas.items()
                    if e.value is not None
                },
                step,
            )

    def log_artifact(
        self, path: os.PathLike[str] | str, artifact_path: str | None = None
    ) -> None:
        """Attach a file (the saved checkpoint, a per-epoch validation report) to this
        run, optionally under an ``artifact_path`` subdir so a family of files (one per
        epoch) groups together. Thread-safe (``MlflowClient``), so the async validator
        worker can call it directly."""
        self._client.log_artifact(self._run_id, str(path), artifact_path=artifact_path)

    def on_epoch_end(self, logs: EpochLogs) -> None:
        # Slash-namespaced so MLflow groups them: loss/<head>/{train,val} sit under a
        # "loss" section, one chart per head (total, onset, voicing) x phase.
        metrics = {_metric_key(h, "train"): v for h, v in logs.train_losses.items()}
        metrics.update({_metric_key(h, "val"): v for h, v in logs.val_losses.items()})
        # Harness scores when the loop ran them inline. Already namespaced
        # (note/<cond>/f1, cues/<cond>/onset_ap) so they pass through untouched -- the
        # async path sends the identical keys straight to `sink`, so a run reads the
        # same whichever way it was scored.
        metrics.update(logs.scores)
        self.sink(metrics, logs.epoch)

    def on_train_end(self, best_epoch: int) -> None:
        self._client.log_metric(self._run_id, "best_epoch", float(best_epoch))
        self._client.set_terminated(self._run_id)
