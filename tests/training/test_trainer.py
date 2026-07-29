"""Trainer wiring: the loss sum, backward, and step actually reduce loss, and
the pitch head is refused until its encoding is settled.

The fit test overfits a single fixed (features -> onset pattern) mapping, so a
working loop must drive the loss down; if forward/loss/backward/step were
miswired it would stay flat.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from stemify.encoders.heads import onset_head
from stemify.encoders.networks import TCN
from stemify.encoders.tagger import FrameTagger
from stemify.features import N_BINS
from stemify.training.config import TrainConfig
from stemify.training.datasets.frame import collate
from stemify.training.losses import FrameBCELoss
from stemify.training.trainer import Trainer


class _FixedMapping(Dataset):
    """The same (features, onset target) repeated -- a thing the model can memorise."""

    def __init__(self, n: int, window: int):
        torch.manual_seed(0)
        self.x = torch.randn(1, N_BINS, window)
        self.y = (torch.rand(window) > 0.7).float()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x.clone(), {"onset": self.y.clone()}


def test_fit_reduces_loss():
    loader = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    model = FrameTagger(TCN(), {"onset": onset_head()})
    config = TrainConfig(
        heads=("onset",), device="cpu", max_epochs=20,
        learning_rate=1e-2, patience=99,
    )
    history = Trainer(model, FrameBCELoss(("onset",)), config).fit(loader)
    assert history.train_loss[-1] < history.train_loss[0]


def test_best_checkpoint_is_restored():
    loader = DataLoader(_FixedMapping(4, window=16), batch_size=4, collate_fn=collate)
    model = FrameTagger(TCN(), {"onset": onset_head()})
    config = TrainConfig(heads=("onset",), device="cpu", max_epochs=5, patience=99)
    history = Trainer(model, FrameBCELoss(("onset",)), config).fit(loader)
    assert 0 <= history.best_epoch < len(history.val_loss)
    # best_epoch tracks the minimum up to the 1e-5 improvement threshold the
    # early-stopper uses, so it is at (or within that threshold of) the minimum.
    assert history.val_loss[history.best_epoch] <= min(history.val_loss) + 1e-5


def test_pitch_head_loss_is_refused():
    with pytest.raises(NotImplementedError):
        FrameBCELoss(["pitch"])


def test_on_batch_fires_per_train_batch_with_monotonic_steps():
    # 8 windows, batch 4 -> 2 train batches/epoch; the val pass must NOT emit.
    train = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    val = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    model = FrameTagger(TCN(), {"onset": onset_head()})
    config = TrainConfig(heads=("onset",), device="cpu", max_epochs=2, patience=99)

    calls: list[tuple[int, dict]] = []

    class _BatchRecorder:
        def on_batch_end(self, step, losses):
            calls.append((step, losses))

        def on_epoch_end(self, logs):
            ...

        def on_train_end(self, best_epoch):
            ...

    Trainer(model, FrameBCELoss(("onset",)), config).fit(train, val, callbacks=[_BatchRecorder()])
    steps = [s for s, _ in calls]
    assert steps == [1, 2, 3, 4]                     # 2 batches x 2 epochs, val excluded
    # per-head dict incl. the total; onset is the one trained head here
    assert all("total" in losses and "onset" in losses for _, losses in calls)

# --- the objective's own parameters must survive snapshot / restore / save ---------
# The Trainer optimises chain(model.parameters(), loss.parameters()), so a snapshot
# that captures only the model silently discards half the trained system. Harmless for
# frame BCE (no parameters) and therefore invisible until a structured objective --
# where the discarded half IS the decoder.


class _ScalarLoss(FrameBCELoss):
    """A frame loss that also owns one learned scalar, standing in for the semi-CRF's
    potentials without pulling the whole CRF into a trainer test."""

    def __init__(self, heads):
        super().__init__(heads, {})
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def parameters(self):
        return [self.scale]

    def compute(self, model, batch, device):
        loss, metrics = super().compute(model, batch, device)
        return loss * self.scale.abs(), metrics

    def state_dict(self):
        return {"scale": self.scale.detach().cpu().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.scale.copy_(state["scale"])

    def to(self, device):
        self.scale = torch.nn.Parameter(self.scale.detach().to(device))
        return self


def _tiny_trainer(loss):
    backbone = TCN()
    model = FrameTagger(backbone, {"onset": onset_head(backbone.hidden)})
    config = TrainConfig(heads=("onset",), max_epochs=3, batch_size=4, device="cpu")
    return Trainer(model, loss, config)


def test_snapshot_captures_both_model_and_objective():
    trainer = _tiny_trainer(_ScalarLoss(["onset"]))
    state = trainer.state_cpu()
    assert set(state) == {"model", "loss"}
    assert "scale" in state["loss"]          # the objective's parameter is captured
    assert state["model"]                    # and the model's still are


def test_parameterless_objective_snapshots_an_empty_loss_state():
    trainer = _tiny_trainer(FrameBCELoss(["onset"], {}))
    assert trainer.state_cpu()["loss"] == {}  # why the frame path never noticed


def test_load_state_restores_the_objectives_parameters():
    trainer = _tiny_trainer(_ScalarLoss(["onset"]))
    snapshot = trainer.state_cpu()
    with torch.no_grad():
        trainer.loss.scale.copy_(torch.tensor(7.5))   # drift, as further epochs would
    trainer.load_state(snapshot)
    assert trainer.loss.scale.item() == pytest.approx(1.0)


def test_fit_restores_the_objective_from_the_best_epoch_not_the_last():
    # The subtle half of the bug: restoring only the model leaves the objective at its
    # final-epoch values, pairing an encoder and a decoder from different points in
    # optimisation. Force epoch 0 to be best, then let later epochs move `scale`.
    loader = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    trainer = _tiny_trainer(_ScalarLoss(["onset"]))

    # A val loss that rises every epoch makes epoch 0 the best; each "train" epoch
    # moves `scale`, standing in for the optimiser walking the potentials away from
    # wherever the best checkpoint was.
    scores = iter([0.1, 0.2, 0.3])

    def fake_epoch(loader_, train, on_batch=None):
        if not train:
            return {"total": next(scores)}
        with torch.no_grad():
            trainer.loss.scale += 1.0
        return {"total": 1.0}

    trainer._run_epoch = fake_epoch
    trainer.fit(loader, loader)

    # epoch 0 trained scale to 2.0 and was best; epochs 1-2 walked it to 4.0.
    assert trainer.loss.scale.item() == pytest.approx(2.0), (
        "restored the objective from the last epoch, not the best one"
    )
