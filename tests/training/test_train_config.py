"""``TrainConfig`` resolves and refuses what follows from its own fields.

The one that matters is ``monitor_mode``. It used to be inferred in
``training/cli.py``, so a config built any other way -- twenty sites do -- kept the
``"min"`` default. Paired with a harness monitor that means selecting on *lowest* F1:
the run finishes, reports success, and hands back its worst checkpoint. This project
has already been bitten by that once, through the other door (a semi-CRF run whose val
loss called the worse epoch better), which is why the inference now holds for every
construction rather than for the one that came through a CLI.
"""

import pytest

from stemify.fitting.config import TrainConfig


@pytest.mark.parametrize("monitor", ["loss/total/val", "loss/onset/val"])
def test_a_loss_monitor_is_minimised(monitor):
    assert TrainConfig(monitor=monitor).monitor_mode == "min"
    assert not TrainConfig(monitor=monitor).maximise


@pytest.mark.parametrize(
    "monitor", ["note/clean/f1", "note/separated/f1", "cues/clean/onset_ap"]
)
def test_a_harness_monitor_is_maximised(monitor):
    """The trap. Every one of these is a score in [0, 1] where higher is better, and
    every one of them defaulted to "min" outside the CLI."""
    assert TrainConfig(monitor=monitor).monitor_mode == "max"
    assert TrainConfig(monitor=monitor).maximise


def test_an_explicit_mode_is_never_overridden():
    """Inference fills a gap; it does not overrule a caller who said what they meant."""
    assert TrainConfig(monitor="note/clean/f1", monitor_mode="min").monitor_mode == "min"
    assert TrainConfig(monitor="loss/total/val", monitor_mode="max").monitor_mode == "max"


def test_a_nonsense_mode_is_refused_at_construction():
    """Not at the first epoch boundary, which is where ``Trainer`` used to catch it."""
    with pytest.raises(ValueError, match="'min' or 'max'"):
        TrainConfig(monitor_mode="lowest")


@pytest.mark.parametrize(
    "field,value",
    [("batch_size", 0), ("max_epochs", 0), ("accumulate_steps", 0), ("patience", 0)],
)
def test_degenerate_counts_are_refused(field, value):
    with pytest.raises(ValueError, match="at least 1"):
        TrainConfig(**{field: value})


def test_a_nonpositive_learning_rate_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        TrainConfig(learning_rate=0.0)
