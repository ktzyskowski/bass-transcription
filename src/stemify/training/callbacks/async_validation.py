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
