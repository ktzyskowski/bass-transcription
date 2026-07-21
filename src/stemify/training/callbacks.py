"""Training callbacks: the extension point the ``Trainer`` core stays free of.

The loop calls ``on_epoch_end`` / ``on_train_end`` with a plain ``EpochLogs``;
everything optional -- experiment tracking, the slow note-level validation --
lives out here as a callback. The dependency is one-directional (trainer imports
callbacks, never the reverse), so nothing here references ``Trainer`` and the
trainer imports no mlflow, staying unit-testable.

Two are provided:

  - ``MLflowCallback`` logs metrics to an MLflow run via ``MlflowClient`` (not the
    global fluent API), so the async validator's worker thread and the main loop
    can log to the same run at once without racing on a shared active run.
  - ``AsyncValidationCallback`` runs an expensive ``eval_fn`` -- the note-level
    harness over a checkpoint -- off the training loop. It snapshots the weights
    to CPU on the main thread, then submits that frozen copy to a background
    worker, so training continues while the checkpoint is scored; the result goes
    to a ``sink`` (e.g. ``MLflowCallback.sink``) when ready. Score on CPU (the
    model is tiny) to keep it off the training device entirely.
"""

from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

Metrics = Mapping[str, float]
StateDict = Mapping[str, Any]


@dataclass(frozen=True)
class EpochLogs:
    """What the trainer hands a callback each epoch -- data, not the trainer.

    ``snapshot`` returns a fresh CPU copy of the model weights when called; the
    async validator calls it on the main thread to freeze a checkpoint before
    handing it off, so the worker never sees weights mutating under it.
    """

    epoch: int
    train_loss: float
    val_loss: float
    best_epoch: int
    snapshot: Callable[[], StateDict]


class Callback:
    """No-op base. Override the hooks you need."""

    def on_epoch_end(self, logs: EpochLogs) -> None:
        ...

    def on_train_end(self, best_epoch: int) -> None:
        ...


class MLflowCallback(Callback):
    """Log train/val loss (and, via ``sink``, async note-level metrics) to MLflow."""

    def __init__(
        self,
        run_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        tracking_uri: str | None = None,
        experiment: str | None = None,
    ) -> None:
        import mlflow  # lazy: tracking is optional, the trainer never imports it
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_id = "0"
        if experiment:
            experiment_id = mlflow.set_experiment(experiment).experiment_id
        self._client = MlflowClient()
        self._run_id = self._client.create_run(experiment_id, run_name=run_name).info.run_id
        for key, value in (params or {}).items():
            self._client.log_param(self._run_id, key, value)

    def sink(self, metrics: Metrics, step: int) -> None:
        """Thread-safe metric logging -- the entry point for the async validator."""
        for key, value in metrics.items():
            self._client.log_metric(self._run_id, key, float(value), step=step)

    def on_epoch_end(self, logs: EpochLogs) -> None:
        self.sink({"train_loss": logs.train_loss, "val_loss": logs.val_loss}, logs.epoch)

    def on_train_end(self, best_epoch: int) -> None:
        self._client.log_metric(self._run_id, "best_epoch", float(best_epoch))
        self._client.set_terminated(self._run_id)


class AsyncValidationCallback(Callback):
    """Score a checkpoint off the training loop and deliver the result to ``sink``.

    ``eval_fn(state_dict, epoch) -> metrics`` is the expensive note-level eval; it
    receives a CPU copy of the weights, so it never touches the live model or the
    training device. Runs every ``every`` epochs; pending jobs are drained at the
    end so no result is lost.
    """

    def __init__(
        self,
        eval_fn: Callable[[StateDict, int], Metrics],
        sink: Callable[[Metrics, int], None],
        every: int = 5,
        executor: Executor | None = None,
    ) -> None:
        self.eval_fn = eval_fn
        self.sink = sink
        self.every = every
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future] = []

    def on_epoch_end(self, logs: EpochLogs) -> None:
        if (logs.epoch + 1) % self.every != 0:
            return
        state = logs.snapshot()  # freeze the checkpoint on the main thread
        future = self._executor.submit(self.eval_fn, state, logs.epoch)
        future.add_done_callback(lambda f, e=logs.epoch: self.sink(f.result(), e))
        self._pending.append(future)

    def on_train_end(self, best_epoch: int) -> None:
        for future in self._pending:
            future.result()  # surface any worker exception, and let sinks fire
        self._executor.shutdown(wait=True)
