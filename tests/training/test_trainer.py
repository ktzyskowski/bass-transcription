"""Trainer wiring: the loss sum, backward, and step actually reduce loss, and
the pitch head is refused until its encoding is settled.

The fit test overfits a single fixed (features -> onset pattern) mapping, so a
working loop must drive the loss down; if forward/loss/backward/step were
miswired it would stay flat.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from stemify.models.heads import onset_head
from stemify.models.networks import TCN
from stemify.models.cuenet import CueNet
from stemify.core.features import N_BINS
from stemify.training.config import TrainConfig
from stemify.training.frame.dataset import collate
from stemify.training.frame.loss import FrameBCELoss
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
    model = CueNet(TCN(), {"onset": onset_head()})
    config = TrainConfig(
        device="cpu", max_epochs=20,
        learning_rate=1e-2, patience=99,
    )
    history = Trainer(model, FrameBCELoss(("onset",)), config).fit(loader)
    assert history.train_loss[-1] < history.train_loss[0]


def test_best_checkpoint_is_restored():
    loader = DataLoader(_FixedMapping(4, window=16), batch_size=4, collate_fn=collate)
    model = CueNet(TCN(), {"onset": onset_head()})
    config = TrainConfig(device="cpu", max_epochs=5, patience=99)
    history = Trainer(model, FrameBCELoss(("onset",)), config).fit(loader)
    assert 0 <= history.best_epoch < len(history.val_loss)
    # best_epoch tracks the minimum up to the 1e-5 improvement threshold the
    # early-stopper uses, so it is at (or within that threshold of) the minimum.
    assert history.val_loss[history.best_epoch] <= min(history.val_loss) + 1e-5


def test_unknown_head_loss_is_refused():
    with pytest.raises(NotImplementedError):
        FrameBCELoss(["tablature"])


def test_pitch_head_is_scored_on_voiced_frames_only():
    """Pitch was refused until its target encoding was settled; it is now the two-hot
    CQT-bin code. Unvoiced frames (NaN targets) must not contribute -- where the bass
    sounds is the voicing head's question."""
    from stemify.core.features import N_BINS
    from stemify.models.heads import pitch_ce

    logits = torch.zeros(2, 4, N_BINS)
    midi = torch.full((2, 4), float("nan"))
    # All-unvoiced: no signal, but still a real tensor in the graph.
    assert float(pitch_ce(logits, midi)) == 0.0

    midi[0, 0] = 45.0
    loss = pitch_ce(logits, midi)
    assert float(loss) > 0.0
    # One voiced frame of four: the mean is over voiced frames, not all frames, so
    # adding more unvoiced frames must not dilute it.
    wider = torch.full((2, 8), float("nan"))
    wider[0, 0] = 45.0
    assert float(pitch_ce(torch.zeros(2, 8, N_BINS), wider)) == pytest.approx(
        float(loss)
    )


def test_pitch_loss_starts_on_the_same_scale_as_the_binary_heads():
    """A uniform head must score ln(n_bins), not n_bins * ln 2.

    This is the whole reason the loss is cross-entropy over the bin axis rather than
    summed per-bin BCE. The first version summed BCE and measured 108.4 against onset
    0.249 and voicing 0.267 -- ~400x, enough to dominate a shared trunk. Asserting the
    initial value pins the *functional form*: summed BCE cannot pass this.
    """
    import math

    from stemify.core.features import N_BINS
    from stemify.models.heads import pitch_ce

    midi = torch.full((2, 4), float("nan"))
    midi[0, 0], midi[1, 2] = 45.0, 38.5   # one on a bin centre, one between two
    uniform = pitch_ce(torch.zeros(2, 4, N_BINS), midi)
    assert float(uniform) == pytest.approx(math.log(N_BINS), abs=1e-4)
    # ...and that is single digits, the scale a BCE term lives on.
    assert float(uniform) < 10.0


def test_on_batch_fires_per_train_batch_with_monotonic_steps():
    # 8 windows, batch 4 -> 2 train batches/epoch; the val pass must NOT emit.
    train = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    val = DataLoader(_FixedMapping(8, window=16), batch_size=4, collate_fn=collate)
    model = CueNet(TCN(), {"onset": onset_head()})
    config = TrainConfig(device="cpu", max_epochs=2, patience=99)

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
    model = CueNet(backbone, {"onset": onset_head(backbone.hidden)})
    config = TrainConfig(max_epochs=3, batch_size=4, device="cpu")
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
