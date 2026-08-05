"""``BestCheckpointCallback``: persist the best checkpoint the moment it exists.

``fit`` restores the best weights when it returns, and ``train`` saves them after --
so a run that finishes, or early-stops, keeps its checkpoint. A run that *stops* does
not. ``KeyboardInterrupt`` propagates straight out of ``fit``, past the save, and a
SIGKILL, an OOM or a dead machine never reach it either. Three semi-CRF runs were
killed on 2026-08-04 after hours of training and left nothing behind, which is the
whole reason this exists.

Writing on improvement rather than at the end also makes the artifact survive failures
the loop cannot catch at all. ``Checkpoint.save`` is atomic, so a kill mid-write leaves
the previous best intact rather than a torn file.
"""

import logging
from pathlib import Path

from stemify.training.callbacks.base import Callback, EpochLogs
from stemify.training.checkpoint import Checkpoint
from stemify.training.recipes import Recipe

logger = logging.getLogger(__name__)


class BestCheckpointCallback(Callback):
    """Write ``path`` whenever the epoch just finished is the new best.

    Best is decided by the trainer, not here: ``logs.best_epoch == logs.epoch`` is the
    loop's own verdict under whatever ``config.monitor`` names, so this cannot drift
    from the criterion early stopping uses. Duplicating the comparison is exactly how
    a checkpoint comes to be chosen by one rule and stopped by another.
    """

    def __init__(self, path: Path, recipe: Recipe) -> None:
        self.path = Path(path)
        self.recipe = recipe
        self.saved_epoch: int | None = None

    def on_epoch_end(self, logs: EpochLogs) -> None:
        if logs.best_epoch != logs.epoch:
            return
        state = logs.snapshot()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        Checkpoint(
            recipe=self.recipe,
            model_state=state["model"],
            objective_state=state["loss"],
        ).save(self.path)
        self.saved_epoch = logs.epoch
        logger.info(
            "saved best checkpoint (epoch %d, %s %.4f) -> %s",
            logs.epoch, logs.monitor or "val loss", logs.monitored, self.path,
        )


class EveryEpochCheckpointCallback(Callback):
    """Write ``<stem>.epoch<N><suffix>`` after every epoch, best or not.

    "Best" is only as good as the metric deciding it, and that metric is a decode --
    so an epoch can be discarded for how it was *read* rather than for what it
    learned. The semi-CRF's Viterbi readout drifts away from its own distribution as
    the model sharpens, which cost three epochs of a run whose losses and cue metrics
    improved throughout: they scored worse, were never written, and could not
    afterwards be re-scored under a corrected decode to find out.

    So this keeps every epoch and lets selection happen offline, where the readout can
    change without re-training. Cheap insurance -- a checkpoint here is single-digit
    megabytes against hours per epoch.
    """

    def __init__(self, path: Path, recipe: Recipe) -> None:
        self.path = Path(path)
        self.recipe = recipe
        self.saved: list[Path] = []

    def on_epoch_end(self, logs: EpochLogs) -> None:
        state = logs.snapshot()
        out = self.path.with_name(
            f"{self.path.stem}.epoch{logs.epoch}{self.path.suffix}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        Checkpoint(
            recipe=self.recipe,
            model_state=state["model"],
            objective_state=state["loss"],
        ).save(out)
        self.saved.append(out)
        logger.info("saved epoch checkpoint -> %s", out)
