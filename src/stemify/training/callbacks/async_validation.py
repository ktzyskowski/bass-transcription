"""``AsyncValidationCallback``: score a checkpoint off the training loop.

The expensive note-level eval (the harness over a checkpoint) runs on a background
worker so training continues while it scores. A CPU snapshot of the weights is frozen
on the main thread before hand-off, so the worker never sees weights mutating under
it, and the model is tiny enough to score on CPU -- keeping it off the training
device entirely. Results go to a ``sink`` (e.g. ``MLflowCallback.sink``) when ready.
"""

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor

from stemify.training.callbacks.base import Callback, EpochLogs, Metrics, StateDict


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
        self._pending: list[Future[Metrics]] = []
        self._delivery_failure: BaseException | None = None

    def _deliver(self, future: Future[Metrics], epoch: int) -> None:
        """Hand a finished score to the sink, keeping any failure for the main thread.

        ``concurrent.futures`` *logs* an exception raised inside a done-callback and
        then discards it, so a sink that rejects what it is handed -- MLflow refusing
        a metric name, a full disk -- would leave training running happily to
        completion, exiting 0, with no validation metrics recorded anywhere and only
        a log line to say so. That is the failure that gets written up as a result.
        """
        try:
            self.sink(future.result(), epoch)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            if self._delivery_failure is None:
                self._delivery_failure = exc

    def _check_delivery(self) -> None:
        """Re-raise a swallowed sink failure, on the main thread where it stops the run."""
        if self._delivery_failure is not None:
            raise RuntimeError(
                "async validation ran but its results were never recorded; the run "
                "would have finished with no validation metrics"
            ) from self._delivery_failure

    def on_epoch_end(self, logs: EpochLogs) -> None:
        # Checked here, not only at the end, so a sink that rejects everything costs
        # one epoch rather than a whole run.
        self._check_delivery()
        if (logs.epoch + 1) % self.every != 0:
            return
        state = logs.snapshot()  # freeze the checkpoint on the main thread
        # Bound to a local before the closure rather than pinned as a lambda default
        # argument. Both capture *this* epoch -- `logs` is re-read next epoch, the local
        # is not -- and the local keeps the lambda's signature inferable.
        epoch = logs.epoch
        future = self._executor.submit(self.eval_fn, state, epoch)
        future.add_done_callback(lambda f: self._deliver(f, epoch))
        self._pending.append(future)

    def on_train_end(self, best_epoch: int) -> None:
        for future in self._pending:
            future.result()  # surface any worker exception, and let sinks fire
        self._executor.shutdown(wait=True)
        self._check_delivery()
