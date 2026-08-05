"""A run's checkpoint path is named deliberately, or nothing is written.

There used to be a default, `<cache-dir>/onset_model.pt`, shared by every run ever
launched without `--out`. It cost two checkpoints whose numbers are still cited in
`docs/experiments.md`: tcn-v1 was overwritten by tcn-v2, a warning was written into the
doc, and tcn-v2 was then overwritten anyway -- by a two-track smoke run, which had no
use for a checkpoint at all.

Both halves of that are covered here. The path is no longer defaulted, so a smoke run
persists nothing; and an occupied path is refused **before the run starts**, because a
job that discovers the collision after hours of training can only clobber or be thrown
away.

These call `train` with a root that does not exist. That is the point: every assertion
below must fire before the first loader is built, so reaching the filesystem at all
would mean the check is too late to be useful.
"""

from pathlib import Path

import pytest

from stemify.training.config import TrainConfig
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.train import train


def _train(tmp_path: Path, **kwargs):
    return train(
        root=str(tmp_path / "no-such-corpus"),
        cache_dir=tmp_path / "cache",
        config=TrainConfig(max_epochs=1),
        recipe=FrameRecipe(conditions=["clean"]),
        **kwargs,
    )


def test_an_occupied_path_is_refused(tmp_path):
    occupied = tmp_path / "rung1-bichannel-pitch.pt"
    occupied.write_bytes(b"a result someone still cites")
    with pytest.raises(FileExistsError, match="already exists"):
        _train(tmp_path, out=occupied)
    assert occupied.read_bytes() == b"a result someone still cites"


def test_force_allows_the_overwrite(tmp_path):
    """`--force` must get *past* the guard -- and then fail on the missing corpus,
    which is how we know the guard was the only thing standing in the way rather than
    the run having quietly stopped for some other reason."""
    occupied = tmp_path / "scratch.pt"
    occupied.write_bytes(b"disposable")
    with pytest.raises(Exception) as excinfo:
        _train(tmp_path, out=occupied, force=True)
    assert not isinstance(excinfo.value, FileExistsError)


def test_no_out_means_nothing_is_written(tmp_path):
    """The smoke-test path. It must not resurrect a default under another name: the
    only evidence that works is an empty directory afterwards."""
    with pytest.raises(Exception) as excinfo:
        _train(tmp_path, out=None)
    assert not isinstance(excinfo.value, FileExistsError)
    written = [p for p in tmp_path.rglob("*.pt")]
    assert written == [], f"a run without --out wrote {written}"


def test_save_epochs_without_out_is_refused(tmp_path):
    """`--save-epochs` derives its filenames from `--out`. Silently writing nothing
    would be the worst outcome: the run asked to keep every epoch so selection could
    happen offline, and would finish having kept none."""
    with pytest.raises(ValueError, match="--save-epochs needs --out"):
        _train(tmp_path, out=None, save_epochs=True)
