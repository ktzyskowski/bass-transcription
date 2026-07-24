"""Callback wiring: the loop invokes hooks with data (never the trainer), the
async validator overlaps eval with training and delivers to a sink, and MLflow
logging round-trips.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from stemify.encoders.heads import onset_head
from stemify.encoders.networks import TCN
from stemify.encoders.tagger import FrameTagger
from stemify.features import N_BINS
from stemify.training.callbacks import (
    AsyncValidationCallback,
    Callback,
    ConsoleCallback,
    EpochLogs,
    MLflowCallback,
)
from stemify.training.config import TrainConfig
from stemify.training.datasets.frame import collate
from stemify.training.losses import FrameBCELoss
from stemify.training.trainer import Trainer


class _FixedMapping(Dataset):
    def __init__(self, n, window):
        torch.manual_seed(0)
        self.x = torch.randn(1, N_BINS, window)
        self.y = (torch.rand(window) > 0.7).float()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x.clone(), {"onset": self.y.clone()}


def _loader(n=4, window=16):
    return DataLoader(_FixedMapping(n, window), batch_size=4, collate_fn=collate)


def _trainer(max_epochs=3):
    model = FrameTagger(TCN(), {"onset": onset_head()})
    return Trainer(model, FrameBCELoss(("onset",)),
                   TrainConfig(heads=("onset",), device="cpu",
                               max_epochs=max_epochs, patience=99))


class _Recorder(Callback):
    def __init__(self):
        self.epochs = []
        self.ended = None

    def on_epoch_end(self, logs: EpochLogs):
        self.epochs.append(logs)

    def on_train_end(self, best_epoch: int):
        self.ended = best_epoch


def test_epoch_hook_receives_data_not_the_trainer():
    rec = _Recorder()
    _trainer(max_epochs=3).fit(_loader(), callbacks=[rec])
    assert [e.epoch for e in rec.epochs] == [0, 1, 2]
    assert all(isinstance(e.train_loss, float) for e in rec.epochs)
    assert rec.ended is not None
    # snapshot is callable and returns weights, so the async validator can freeze one
    assert callable(rec.epochs[0].snapshot)
    assert dict(rec.epochs[0].snapshot())  # a populated state_dict -- real weights


def test_console_callback_logs_progress(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="stemify.training.callbacks"):
        _trainer(max_epochs=2).fit(_loader(), callbacks=[ConsoleCallback()])
    assert "epoch" in caplog.text
    assert "training done" in caplog.text


def test_async_validation_delivers_to_sink():
    delivered = {}

    def eval_fn(state_dict, epoch):
        assert state_dict  # got real weights (a populated state_dict), not a placeholder
        return {"note_f1": 0.5 + epoch}

    def sink(metrics, step):
        delivered[step] = metrics["note_f1"]

    cb = AsyncValidationCallback(
        eval_fn, sink, every=1, executor=ThreadPoolExecutor(max_workers=1)
    )
    _trainer(max_epochs=3).fit(_loader(), callbacks=[cb])  # on_train_end drains
    assert delivered == {0: 0.5, 1: 1.5, 2: 2.5}


def test_async_validation_respects_every():
    calls = []
    cb = AsyncValidationCallback(
        lambda s, e: calls.append(e) or {"m": 0.0}, lambda m, s: None, every=2,
    )
    _trainer(max_epochs=4).fit(_loader(), callbacks=[cb])
    assert calls == [1, 3]  # epochs 1 and 3 (every 2nd, 0-indexed +1)


def test_mlflow_callback_logs_metrics(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"file://{tmp_path}/mlruns"
    cb = MLflowCallback(run_name="t", params={"lr": 0.01}, tracking_uri=uri)
    _trainer(max_epochs=3).fit(_loader(), callbacks=[cb])

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(cb._run_id)
    assert run.data.params["lr"] == "0.01"
    history = client.get_metric_history(cb._run_id, "loss/total/train")
    assert len(history) == 3
    assert client.get_metric_history(cb._run_id, "loss/onset/train")  # per-head logged
    assert run.data.metrics["best_epoch"] >= 0


def test_mlflow_callback_logs_smoothed_loss_on_throttle_boundary(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"file://{tmp_path}/mlruns"
    cb = MLflowCallback(run_name="t", tracking_uri=uri, batch_log_every=2, ema_beta=0.5)
    for step, loss in enumerate([1.0, 3.0, 5.0, 7.0], start=1):
        cb.on_batch_end(step, {"total": loss})  # EMA: 1, 2, 3.5, 5.25; logged at 2 and 4

    client = MlflowClient(tracking_uri=uri)
    history = client.get_metric_history(cb._run_id, "loss/total/step")
    by_step = {m.step: m.value for m in history}
    assert set(by_step) == {2, 4}                 # only the throttle boundaries
    assert by_step[2] == pytest.approx(2.0)       # smoothed, not the raw 3.0
    assert by_step[4] == pytest.approx(5.25)


def test_mlflow_batch_logging_disabled_when_zero(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"file://{tmp_path}/mlruns"
    cb = MLflowCallback(run_name="t", tracking_uri=uri, batch_log_every=0)
    for step in range(1, 6):
        cb.on_batch_end(step, {"total": 1.0})

    client = MlflowClient(tracking_uri=uri)
    assert client.get_metric_history(cb._run_id, "loss/total/step") == []


def test_mlflow_callback_logs_artifact(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    uri = f"file://{tmp_path}/mlruns"
    cb = MLflowCallback(run_name="t", tracking_uri=uri)
    ckpt = tmp_path / "onset_model.pt"
    ckpt.write_bytes(b"weights")
    cb.log_artifact(ckpt)

    client = MlflowClient(tracking_uri=uri)
    artifacts = [a.path for a in client.list_artifacts(cb._run_id)]
    assert "onset_model.pt" in artifacts