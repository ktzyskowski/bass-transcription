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
from stemify.training.dataset import collate
from stemify.training.trainer import Trainer, build_losses


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
    history = Trainer(model, config).fit(loader)
    assert history.train_loss[-1] < history.train_loss[0]


def test_best_checkpoint_is_restored():
    loader = DataLoader(_FixedMapping(4, window=16), batch_size=4, collate_fn=collate)
    model = FrameTagger(TCN(), {"onset": onset_head()})
    config = TrainConfig(heads=("onset",), device="cpu", max_epochs=5, patience=99)
    history = Trainer(model, config).fit(loader)
    assert 0 <= history.best_epoch < len(history.val_loss)
    # best_epoch tracks the minimum up to the 1e-5 improvement threshold the
    # early-stopper uses, so it is at (or within that threshold of) the minimum.
    assert history.val_loss[history.best_epoch] <= min(history.val_loss) + 1e-5


def test_pitch_head_loss_is_refused():
    with pytest.raises(NotImplementedError):
        build_losses(["pitch"], pos_weight={}, device=torch.device("cpu"))


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

    Trainer(model, config).fit(train, val, callbacks=[_BatchRecorder()])
    steps = [s for s, _ in calls]
    assert steps == [1, 2, 3, 4]                     # 2 batches x 2 epochs, val excluded
    # per-head dict incl. the total; onset is the one trained head here
    assert all("total" in losses and "onset" in losses for _, losses in calls)