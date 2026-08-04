"""The callback contract: the extension point the ``Trainer`` core stays free of.

The loop calls ``on_epoch_end`` / ``on_train_end`` with a plain ``EpochLogs``;
everything optional -- experiment tracking, the slow note-level validation -- lives
in a sibling module as a ``Callback`` subclass. The dependency is one-directional
(trainer imports callbacks, never the reverse), so nothing here references ``Trainer``
and the trainer imports no mlflow, staying unit-testable.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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
    train_loss: float                    # the total (sum over heads), for early stop
    val_loss: float
    train_losses: Mapping[str, float]    # {"total": ..., <head>: ...}, per head
    val_losses: Mapping[str, float]
    best_epoch: int
    snapshot: Callable[[], StateDict]
    # Note-level harness metrics, when the loop scored synchronously (empty when the
    # scoring is async -- that path delivers straight to its own sink instead).
    scores: Mapping[str, float] = field(default_factory=dict)
    # The value early stopping actually compared, and under which key. Reported so a
    # log line can say what drove the decision rather than leaving it to be inferred.
    monitored: float = float("nan")
    monitor: str = ""


class Callback:
    """No-op base. Override the hooks you need."""

    def on_batch_end(self, step: int, losses: Mapping[str, float]) -> None:
        ...

    def on_epoch_end(self, logs: EpochLogs) -> None:
        ...

    def on_train_end(self, best_epoch: int) -> None:
        ...
