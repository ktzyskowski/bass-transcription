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

from stemify.training.callbacks.base import Callback, EpochLogs, Metrics
from stemify.training.ema import EMA

logger = logging.getLogger(__name__)


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
        smoothed = {
            h: self._emas.setdefault(h, EMA(self._ema_beta)).update(v)
            for h, v in losses.items()
        }
        if self._batch_log_every and step % self._batch_log_every == 0:
            self.sink({f"loss/{h}/step": v for h, v in smoothed.items()}, step)

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
        metrics = {f"loss/{h}/train": v for h, v in logs.train_losses.items()}
        metrics.update({f"loss/{h}/val": v for h, v in logs.val_losses.items()})
        self.sink(metrics, logs.epoch)

    def on_train_end(self, best_epoch: int) -> None:
        self._client.log_metric(self._run_id, "best_epoch", float(best_epoch))
        self._client.set_terminated(self._run_id)
