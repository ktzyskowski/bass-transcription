"""Empty loaders must fail where they are empty, not several frames later.

This is a *class* of bug rather than one bug, and the class has now cost two debugging
sessions. Its shape: a component handles a degenerate input by quietly producing
nothing, the emptiness travels, and the failure surfaces somewhere with no connection
to the cause -- or does not surface at all.

The instance these cover: a validation split smaller than ``batch_size`` left
``ChunkDataset.__iter__`` with fewer tracks than streams, where it returned silently.
``Trainer._run_epoch`` then averaged over zero batches and returned ``{}``, and ``fit``
died on ``val_losses["total"]`` -- a ``KeyError`` naming a metric, for a problem about
a dataset. Both halves are covered here, because either alone leaves the other silent.

The sibling instance, kept in mind and covered elsewhere: ``_epoch_batches`` assumed
every loader is sized, which crashed at the first MLflow write while every smoke test
-- run without a tracking URI -- passed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stemify.semicrf import SemiCRFDecoder
from stemify.core.notes import NoteEvent
from stemify.core.serde import CachedItem
from stemify.training.config import TrainConfig
from stemify.training.frame.dataset import FrameBatch
from stemify.training.frame.loss import FrameBCELoss
from stemify.semicrf.fit.dataset import ChunkDataset
from stemify.semicrf.fit.tracks import TrainItem
from stemify.training.trainer import Trainer

N_BINS, HOP_S = 12, 0.0116
N_CQT = 600


class _Tagger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.onset = nn.Linear(N_BINS, 1)

    def forward(self, x):
        return {"onset": self.onset(x.squeeze(1).transpose(1, 2))}


def _frame_batch(b: int = 2, w: int = 8) -> FrameBatch:
    return FrameBatch(
        x=torch.randn(b, 1, N_BINS, w),
        y={"onset": torch.zeros(b, w)},
    )


def _trainer(max_epochs: int = 1) -> Trainer:
    # One epoch by default: TrainConfig's own default is 50, and none of these tests
    # are about convergence.
    return Trainer(
        _Tagger(),
        FrameBCELoss(("onset",)),
        TrainConfig(device="cpu", max_epochs=max_epochs),
    )


# --------------------------------------------------------------------------- #
# Trainer: the backstop, for any loader that yields nothing                    #
# --------------------------------------------------------------------------- #
def test_an_empty_train_loader_names_itself():
    with pytest.raises(ValueError, match="train loader yielded no batches"):
        _trainer().fit([])


def test_an_empty_val_loader_names_itself():
    # The reported instance: training data was fine, the *val* split was too small,
    # and the error arrived as KeyError('total') from inside fit.
    with pytest.raises(ValueError, match="validation loader yielded no batches"):
        _trainer().fit([_frame_batch()], [])


def test_the_error_points_at_the_limit_flags():
    # The message has to carry the fix, not just the fact -- the two reachable causes
    # are an empty split and a limit below the batch size.
    with pytest.raises(ValueError, match="--limit/--val-limit"):
        _trainer().fit([])


def test_a_non_empty_loader_still_averages_normally():
    # The guard must not have changed the ordinary path: two batches, mean over them.
    history = _trainer().fit([_frame_batch(), _frame_batch()])
    assert len(history.train_loss) == 1
    assert np.isfinite(history.train_loss[0])


# --------------------------------------------------------------------------- #
# ChunkDataset: the precise diagnosis, where the emptiness originates          #
# --------------------------------------------------------------------------- #
def _item(tmp: Path, name: str) -> TrainItem:
    path = tmp / f"{name}.npz"
    CachedItem(
        name,
        np.zeros((N_BINS, N_CQT), dtype=np.float32),
        {},
        np.arange(N_CQT) * HOP_S,
    ).save(path)
    return TrainItem(
        features=(path,),
        notes=[NoteEvent(onset=0.1, offset=0.4, midi_pitch=40.0)],
    )


def _chunks(items, batch_size: int) -> ChunkDataset:
    return ChunkDataset(
        items, chunk_frames=40, dmax=30, stride=3, batch_size=batch_size
    )


def test_fewer_tracks_than_streams_is_refused_with_both_numbers(tmp_path):
    items = [_item(tmp_path, f"t{k}") for k in range(2)]
    with pytest.raises(ValueError, match="2 track.* cannot fill 4 parallel streams"):
        list(_chunks(items, batch_size=4))


def test_the_refusal_says_how_to_fix_it(tmp_path):
    items = [_item(tmp_path, f"t{k}") for k in range(3)]
    with pytest.raises(ValueError, match="at most 3"):
        list(_chunks(items, batch_size=8))


def test_exactly_enough_tracks_is_fine(tmp_path):
    # The boundary: one track per stream is the minimum that works, and must.
    items = [_item(tmp_path, f"t{k}") for k in range(4)]
    batches = list(_chunks(items, batch_size=4))
    assert batches and all(len(b) == 4 for b in batches)


def test_the_semicrf_loss_reaches_the_trainer_guard_too(tmp_path):
    """The two guards compose: an over-wide batch size fails at the dataset.

    Worth pinning because the dataset's own refusal is what stops the trainer from
    ever seeing an empty epoch on this path -- if the dataset ever went back to
    returning silently, this test would keep passing while `test_an_empty_val_loader`
    became the only thing standing between a small val set and a KeyError.
    """
    from stemify.semicrf.fit.loss import SemiCRFLoss

    items = [_item(tmp_path, "only")]
    trainer = Trainer(
        _Tagger(), SemiCRFLoss(SemiCRFDecoder(stride=3)), TrainConfig(device="cpu")
    )
    with pytest.raises(ValueError, match="cannot fill"):
        trainer.fit(_chunks(items, batch_size=2))
