"""``ConsoleCallback``: the minimal progress signal for a run -- epoch losses to the log."""

import logging

from stemify.training.callbacks.base import Callback, EpochLogs

logger = logging.getLogger(__name__)


class ConsoleCallback(Callback):
    """Log epoch losses -- the minimal progress signal for a run."""

    def on_epoch_end(self, logs: EpochLogs) -> None:
        # Break out the per-head train loss (onset/voicing/...) beside the total, so
        # the log shows which head is still moving; total drives early stopping. The
        # loop's grad_norm rides along in the same mapping and is worth seeing here for
        # the same reason -- it is the term that moves first when a run is about to go.
        heads = " ".join(
            f"{h} {v:.4f}" for h, v in logs.train_losses.items() if h != "total"
        )
        logger.info(
            "epoch %3d  train %.4f  val %.4f  [%s]  (best %d)",
            logs.epoch, logs.train_loss, logs.val_loss, heads, logs.best_epoch,
        )

    def on_train_end(self, best_epoch: int) -> None:
        logger.info("training done -- best epoch %d", best_epoch)
