"""Two failures that both end with a run reporting success and leaving the wrong model.

**Selecting on the wrong number.** `fit` monitored `val_losses["total"]` and nothing
else. A semi-CRF run scored note F1 0.8175 at epoch 0 and 0.4086 at epoch 1 while its
val loss went 20.04 -> 16.25 and called epoch 1 better; finishing there would have
restored the worse model and said "training done". The deployment metric is
`note/separated/f1`, and monitoring it means scoring on the critical path -- the async
validator delivers epoch N's score partway through N+1, after N's decision is made.

**Losing it anyway.** `fit` restores the best weights when it *returns*, and the save
happens after that, so KeyboardInterrupt/SIGKILL/OOM kept nothing. Three runs died that
way on 2026-08-04 with hours of training in them.
"""

import pytest
import torch
from torch import nn

from stemify.training.base import Loss
from stemify.training.callbacks import (
    BestCheckpointCallback,
    EveryEpochCheckpointCallback,
)
from stemify.training.checkpoint import Checkpoint
from stemify.training.config import TrainConfig
from stemify.training.recipes import RECIPES
from stemify.training.trainer import Trainer


class _Batch(list[int]):
    pass


class _FixedLoss(Loss[_Batch]):
    """Reports a scripted val loss per epoch, so the monitored series is exact."""

    def __init__(self, per_epoch: list[float]) -> None:
        self.w = nn.Parameter(torch.tensor(1.0))
        self.per_epoch = per_epoch
        self.calls = 0

    def compute(self, model, batch, device):
        # One train + one val batch per epoch; both report the epoch's scripted value.
        value = self.per_epoch[min(self.calls // 2, len(self.per_epoch) - 1)]
        self.calls += 1
        loss = self.w * value
        return loss, {"total": float(loss.item())}

    def parameters(self):
        return [self.w]

    def state_dict(self):
        return {"w": self.w.detach().cpu().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.w.copy_(state["w"])


def _trainer(loss, monitor="loss/total/val", mode="min", epochs=3, patience=99):
    return Trainer(
        nn.Linear(1, 1), loss,
        TrainConfig(device="cpu", max_epochs=epochs, patience=patience,
                    monitor=monitor, monitor_mode=mode),
    )


def _batches(n=1):
    return [_Batch([0]) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Selecting on a harness metric                                                #
# --------------------------------------------------------------------------- #
def test_a_harness_metric_can_pick_the_checkpoint():
    """The real case: val loss falls every epoch while F1 collapses. The run must keep
    epoch 0, which is precisely what monitoring the loss would not do."""
    loss = _FixedLoss([20.04, 16.25, 14.0])          # loss says "better" every epoch
    f1 = [0.8175, 0.4086, 0.3000]                     # F1 says epoch 0
    trainer = _trainer(loss, monitor="note/separated/f1", mode="max")
    hist = trainer.fit(_batches(), _batches(),
                       score_fn=lambda state, ep: {"note/separated/f1": f1[ep]})
    assert hist.best_epoch == 0


def test_monitoring_the_loss_would_have_picked_the_worse_one():
    """The contrast, so the test above cannot pass by accident."""
    loss = _FixedLoss([20.04, 16.25, 14.0])
    hist = _trainer(loss).fit(_batches(), _batches())
    assert hist.best_epoch == 2


def test_max_mode_is_not_just_min_with_a_sign():
    loss = _FixedLoss([1.0, 1.0, 1.0])
    f1 = [0.1, 0.9, 0.5]
    hist = _trainer(loss, monitor="note/clean/f1", mode="max").fit(
        _batches(), _batches(), score_fn=lambda s, ep: {"note/clean/f1": f1[ep]})
    assert hist.best_epoch == 1


def test_an_unknown_monitor_key_names_what_is_available():
    """Reachable by a typo, or by naming a harness metric with no score_fn wired. The
    silent alternative -- falling back to the loss -- would select on a different
    quantity than the run claims to."""
    with pytest.raises(KeyError, match="Available:"):
        _trainer(_FixedLoss([1.0]), monitor="note/separated/f1", mode="max").fit(
            _batches(), _batches())


def test_a_bad_mode_is_refused_before_training():
    with pytest.raises(ValueError, match="monitor_mode"):
        _trainer(_FixedLoss([1.0]), mode="maximise").fit(_batches(), _batches())


def test_early_stopping_uses_the_monitored_metric_too():
    """Patience must count on the same number selection uses, or a run stops on one
    criterion and keeps a checkpoint chosen by another."""
    loss = _FixedLoss([5.0] * 6)
    f1 = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]      # best at epoch 0, then flat
    hist = _trainer(loss, monitor="note/clean/f1", mode="max", epochs=6,
                    patience=2).fit(
        _batches(), _batches(), score_fn=lambda s, ep: {"note/clean/f1": f1[ep]})
    assert hist.best_epoch == 0
    assert len(hist.train_loss) == 3, "should stop 2 epochs after the last improvement"


# --------------------------------------------------------------------------- #
# Surviving an interrupt                                                       #
# --------------------------------------------------------------------------- #
def test_the_best_checkpoint_is_on_disk_before_the_run_ends(tmp_path):
    """The interrupt case. `fit` restores the best weights when it *returns*; a
    Ctrl-C never gets there. Written on improvement, the file already exists."""
    recipe = RECIPES["semicrf"](["clean"], pitch=True)
    model, objective = recipe.build_model(), recipe.build_loss()
    out = tmp_path / "best.pt"
    cb = BestCheckpointCallback(out, recipe)

    from stemify.training.callbacks.base import EpochLogs
    state = {"model": model.state_dict(), "loss": objective.state_dict()}
    cb.on_epoch_end(EpochLogs(
        epoch=0, train_loss=1.0, val_loss=1.0, train_losses={"total": 1.0},
        val_losses={"total": 1.0}, best_epoch=0, snapshot=lambda: state,
        monitored=0.8, monitor="note/separated/f1",
    ))
    assert out.exists(), "nothing on disk; an interrupt here would lose the run"
    assert Checkpoint.load(out).recipe.pitch is True


def test_a_non_improving_epoch_does_not_overwrite_the_best(tmp_path):
    """Otherwise the file tracks the *last* epoch, not the best -- which is the bug
    this callback exists to avoid, wearing a different hat."""
    recipe = RECIPES["semicrf"](["clean"], pitch=True)
    model, objective = recipe.build_model(), recipe.build_loss()
    out = tmp_path / "best.pt"
    cb = BestCheckpointCallback(out, recipe)
    from stemify.training.callbacks.base import EpochLogs

    def logs(epoch, best):
        return EpochLogs(
            epoch=epoch, train_loss=1.0, val_loss=1.0, train_losses={"total": 1.0},
            val_losses={"total": 1.0}, best_epoch=best,
            snapshot=lambda: {"model": model.state_dict(),
                              "loss": objective.state_dict()},
        )

    cb.on_epoch_end(logs(0, 0))
    assert cb.saved_epoch == 0
    cb.on_epoch_end(logs(1, 0))          # epoch 1 is not the best
    assert cb.saved_epoch == 0, "overwrote the best with a worse epoch"


def test_every_epoch_is_kept_under_its_own_name(tmp_path):
    """"Best" is decided by a decode, and a decode can be wrong about which epoch was
    better -- the semi-CRF's Viterbi readout drifts from its own distribution as the
    model sharpens, so three improving epochs scored worse, were never written, and
    could not be re-scored afterwards. Keeping every epoch moves selection offline,
    where the readout can change without re-training."""
    recipe = RECIPES["semicrf"](["clean"], pitch=True)
    model, objective = recipe.build_model(), recipe.build_loss()
    cb = EveryEpochCheckpointCallback(tmp_path / "run.pt", recipe)
    from stemify.training.callbacks.base import EpochLogs

    for epoch in range(3):
        cb.on_epoch_end(EpochLogs(
            epoch=epoch, train_loss=1.0, val_loss=1.0, train_losses={"total": 1.0},
            val_losses={"total": 1.0}, best_epoch=0,   # only epoch 0 is ever "best"
            snapshot=lambda: {"model": model.state_dict(),
                              "loss": objective.state_dict()},
        ))

    assert [p.name for p in cb.saved] == [
        "run.epoch0.pt", "run.epoch1.pt", "run.epoch2.pt"
    ]
    # Each must be independently loadable, or "we kept them" is not the same claim as
    # "we can score them".
    for path in cb.saved:
        assert Checkpoint.load(path).recipe.pitch is True
    assert not (tmp_path / "run.pt").exists(), "must not also claim the best-checkpoint path"
