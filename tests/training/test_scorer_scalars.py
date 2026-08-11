"""``Loss.scalars``: the objective's learned potentials, on their own axes.

The semi-CRF trades segmentation *granularity* through learned potentials -- a
duration embedding, a transition prior, the gate on the pitch head -- and none of them
appear in the loss breakdown. A run whose decode goes from roughly the right note count
to two thirds of it, while every loss term and every cue metric improves, is explained
by those numbers and by nothing else that is logged. Before this they existed only
inside a checkpoint, and only for the epochs that happened to be best, so the epoch that
changed character was exactly the epoch whose potentials were never written down.

These pin the plumbing: the objective names them, the loop carries them, the tracking
callback files them apart from the losses.
"""

import pytest
import torch
from torch import nn

from stemify.fitting.base import Loss
from stemify.training.callbacks import Callback, EpochLogs, MLflowCallback
from stemify.fitting.config import TrainConfig
from stemify.training.frame.loss import FrameBCELoss
from stemify.training.recipes import RECIPES
from stemify.training.trainer import Trainer


def _loss():
    return RECIPES["semicrf"](["clean"], pitch=True).build_loss()


# --------------------------------------------------------------------------- #
# What the objective reports                                                   #
# --------------------------------------------------------------------------- #
def test_every_learned_potential_is_named():
    """Including the ones that decide granularity. A subset would be worse than
    nothing: reading the evidence projection alone and concluding "the model is stable"
    says nothing about the transition table that actually moved."""
    scalars = _loss().scalars()
    # A one-element potential lands on its own axis, under its own name.
    assert "scorer/rest_logprob" in scalars
    # Vector-valued potentials expand per element, or the profile that flips is
    # averaged into a single number that barely moves.
    assert {"scorer/class_trans/0", "scorer/class_trans/5"} <= set(scalars)
    assert "scorer/alpha/bias" in scalars
    # Anything wider is summarised rather than dropped -- 49 label biases would
    # otherwise be 49 axes nobody reads.
    assert {"scorer/label_bias/mean", "scorer/label_bias/std"} <= set(scalars)


def test_a_small_potential_is_reported_element_by_element():
    """The failure this exists for: a transition prior favouring note->note flipping to
    favour note->rest. That is a change of *sign pattern* across entries, invisible in
    any summary of them."""
    objective = _loss()
    profile = [-0.24, 0.23, 0.27, 0.29, -0.29, -0.46]
    with torch.no_grad():
        objective.decoder.scorer.class_trans.copy_(
            torch.tensor(profile).reshape(3, 2)
        )
    scalars = objective.scalars()
    got = [scalars[f"scorer/class_trans/{i}"] for i in range(6)]
    assert got == pytest.approx(profile, abs=1e-6)


def test_values_follow_the_live_parameters():
    """Read through to the tensors, not a copy taken at construction -- otherwise every
    epoch reports the initialisation and the series is flat by construction."""
    objective = _loss()
    before = objective.scalars()["scorer/rest_logprob"]
    with torch.no_grad():
        objective.decoder.scorer.rest_logprob += 1.5
    assert objective.scalars()["scorer/rest_logprob"] == pytest.approx(before + 1.5)


def test_a_wide_potential_is_summarised_rather_than_dropped():
    """A neural scorer would swamp the run with one metric per weight; silence would
    read as "unchanged". Neither -- it reports mean and std."""
    objective = _loss()
    objective.decoder.scorer.big = nn.Parameter(torch.ones(64) * 3.0)
    scalars = objective.scalars()
    assert scalars["scorer/big/mean"] == pytest.approx(3.0)
    assert "scorer/big/std" in scalars
    assert "scorer/big/0" not in scalars


def test_an_objective_without_potentials_reports_none():
    """The default has to be empty rather than something: a frame run has no
    interpretable learned scalars, and inventing axes for it would put an always-flat
    chart beside the ones worth reading."""
    assert FrameBCELoss(("onset",)).scalars() == {}


# --------------------------------------------------------------------------- #
# What the loop and the tracking callback do with them                         #
# --------------------------------------------------------------------------- #
class _Batch(list[int]):
    pass


class _ScalarLoss(Loss[_Batch]):
    """Minimal objective with one potential, so the wiring is tested without a DP."""

    def __init__(self) -> None:
        self.w = nn.Parameter(torch.tensor(1.0))

    def compute(self, model, batch, device):
        loss = self.w * 1.0
        return loss, {"total": float(loss.item())}

    def parameters(self):
        return [self.w]

    def scalars(self):
        return {"scorer/b_note": float(self.w.detach())}


class _Recorder(Callback):
    def __init__(self) -> None:
        self.epochs: list[EpochLogs] = []

    def on_epoch_end(self, logs: EpochLogs) -> None:
        self.epochs.append(logs)


def test_the_loop_carries_them_to_callbacks_each_epoch():
    rec = _Recorder()
    Trainer(
        nn.Linear(1, 1), _ScalarLoss(),
        TrainConfig(device="cpu", max_epochs=2, patience=99),
    ).fit([_Batch([0])], [_Batch([0])], callbacks=[rec])
    assert [dict(e.scalars) for e in rec.epochs] != [{}, {}]
    assert all("scorer/b_note" in e.scalars for e in rec.epochs)


def test_they_are_not_filed_under_loss(tmp_path):
    """Same reasoning as the gradient norm: a potential on the ``loss/`` axis is read
    as a loss term, and these are routinely negative."""
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"file://{tmp_path}/mlruns"
    cb = MLflowCallback(run_name="t", tracking_uri=uri)
    cb.on_epoch_end(EpochLogs(
        epoch=0, train_loss=1.0, val_loss=1.0, train_losses={"total": 1.0},
        val_losses={"total": 1.0}, best_epoch=0, snapshot=dict,
        scalars={"scorer/b_note": -0.19, "scorer/dur/0": -0.24},
    ))

    client = MlflowClient(tracking_uri=uri)
    assert client.get_metric_history(cb._run_id, "scalars/scorer/b_note")
    assert client.get_metric_history(cb._run_id, "scalars/scorer/dur/0")
    assert client.get_metric_history(cb._run_id, "loss/scorer/b_note") == []
