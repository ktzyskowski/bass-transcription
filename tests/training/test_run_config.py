"""``RunBuilder``: every setting refused at the point it is named.

These constraints all existed before, as a 140-line wall of ``if``s at the top of
``train()``. What is new is *where* they live and therefore what they can say: a
per-field check names the flag that caused it, and the four pair-wise ones run at
``build()`` -- which is the first moment the whole run exists, and still before the
corpus is indexed or a single feature is computed.

The property under test throughout is **when**, not just whether. Every case here
names a corpus root that does not exist, so anything that reaches the filesystem would
fail for the wrong reason.
"""

import dataclasses
from pathlib import Path

import pytest

from stemify.fitting.config import TrainConfig
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.run import RunBuilder, RunConfig

NOWHERE = "/no-such-corpus"


def _builder(tmp_path: Path, channels: int = 1) -> RunBuilder:
    conditions = ["clean"] if channels == 1 else ["bichannel-clean"]
    return (
        RunBuilder(NOWHERE, tmp_path)
        .with_recipe(FrameRecipe(conditions=conditions))
        .fitting(TrainConfig())
    )


# --------------------------------------------------------------------------- #
# What a setter can refuse on its own                                          #
# --------------------------------------------------------------------------- #
def test_a_zero_limit_is_refused(tmp_path):
    """It trains on nothing and reports a finished run -- the shape of failure this
    project keeps paying for."""
    with pytest.raises(ValueError, match="no tracks at all"):
        _builder(tmp_path).limited_to(0)


def test_a_zero_val_limit_is_refused(tmp_path):
    with pytest.raises(ValueError, match="scores no tracks"):
        _builder(tmp_path).validating(limit=0)


def test_a_zero_val_every_is_refused(tmp_path):
    with pytest.raises(ValueError, match="never validates"):
        _builder(tmp_path).validating(every=0)


def test_a_missing_warm_start_is_refused(tmp_path):
    """A warm start is usually the whole point of the run. A typo'd path must not
    quietly become a from-scratch run discovered hours later."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _builder(tmp_path).warm_starting_from(tmp_path / "not-there.pt")


def test_an_existing_warm_start_is_accepted(tmp_path):
    ckpt = tmp_path / "there.pt"
    ckpt.write_bytes(b"")
    assert _builder(tmp_path).warm_starting_from(ckpt).build().init_from == ckpt


def test_no_warm_start_is_fine(tmp_path):
    assert _builder(tmp_path).warm_starting_from(None).build().init_from is None


def test_a_zero_batch_log_every_is_refused(tmp_path):
    with pytest.raises(ValueError, match="minimum is 1"):
        _builder(tmp_path).tracking("file:./mlruns", batch_log_every=0)


# --------------------------------------------------------------------------- #
# What only build() can see: two fields at once                                #
# --------------------------------------------------------------------------- #
def test_val_conditions_default_to_the_recipes_channel_width(tmp_path):
    """Not knowable in ``validating()`` -- the recipe may be named after it."""
    assert _builder(tmp_path, channels=1).build().val_conditions == [
        "clean", "separated",
    ]
    assert _builder(tmp_path, channels=2).build().val_conditions == [
        "bichannel-clean", "bichannel-separated",
    ]


def test_the_default_holds_whatever_order_the_setters_ran_in(tmp_path):
    """The reason the pair-wise checks are not in setters. Naming the recipe last must
    give the same run as naming it first."""
    late = (
        RunBuilder(NOWHERE, tmp_path)
        .fitting(TrainConfig())
        .with_recipe(FrameRecipe(conditions=["bichannel-clean"]))
        .build()
    )
    assert late.val_conditions == ["bichannel-clean", "bichannel-separated"]


def test_a_val_condition_of_the_wrong_channel_width_is_refused(tmp_path):
    """Scoring needs the layout the model was trained with; a 1-channel model handed
    a bichannel condition fails in the first conv, minutes in."""
    with pytest.raises(ValueError, match="do not match the model"):
        _builder(tmp_path, channels=1).validating(["bichannel-clean"]).build()


def test_a_harness_monitor_requires_validating_every_epoch(tmp_path):
    """On a skipped epoch the monitored metric would not exist and the run would die
    mid-training."""
    builder = (
        RunBuilder(NOWHERE, tmp_path)
        .with_recipe(FrameRecipe(conditions=["clean"]))
        .fitting(TrainConfig(monitor="note/clean/f1", monitor_mode="max"))
        .validating(every=5)
    )
    with pytest.raises(ValueError, match="every epoch"):
        builder.build()


def test_a_misspelled_harness_monitor_is_refused_before_any_work(tmp_path):
    """The check that used to happen at the end of epoch 0 -- past the val feature
    caching and a full pass over the corpus -- and only because ``Trainer`` raised on
    a dict lookup. The names come from the metric result types, so nothing is scored
    to know them."""
    builder = (
        RunBuilder(NOWHERE, tmp_path)
        .with_recipe(FrameRecipe(conditions=["clean"]))
        .fitting(TrainConfig(monitor="note/clean/F1", monitor_mode="max"))
        .validating(every=1)
    )
    with pytest.raises(ValueError, match="not a metric this run will produce"):
        builder.build()


def test_a_harness_monitor_this_run_does_produce_is_accepted(tmp_path):
    run = (
        RunBuilder(NOWHERE, tmp_path)
        .with_recipe(FrameRecipe(conditions=["clean"]))
        .fitting(TrainConfig(monitor="note/separated/f1", monitor_mode="max"))
        .validating(every=1)
        .build()
    )
    assert run.sync_scoring


def test_a_loss_monitor_needs_no_harness_at_all(tmp_path):
    """The cheap default: an inline val loss exists every epoch by construction, so
    neither pair-wise check applies."""
    run = _builder(tmp_path).validating(every=5).build()
    assert not run.sync_scoring


# --------------------------------------------------------------------------- #
# The shape of the result                                                      #
# --------------------------------------------------------------------------- #
def test_build_refuses_an_incomplete_builder(tmp_path):
    with pytest.raises(ValueError, match="no recipe"):
        RunBuilder(NOWHERE, tmp_path).fitting(TrainConfig()).build()
    with pytest.raises(ValueError, match="no TrainConfig"):
        RunBuilder(NOWHERE, tmp_path).with_recipe(FrameRecipe(["clean"])).build()


def test_the_run_is_frozen(tmp_path):
    """A field reassigned after ``build`` would be a run whose settings were checked
    in a state it never ran in."""
    run = _builder(tmp_path).build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.val_every = 9  # type: ignore[misc]


def test_the_run_carries_its_fit_config(tmp_path):
    """One argument to ``train``. The two configs answer different questions -- what to
    run against how to fit it -- but a run is not specified without both."""
    run = _builder(tmp_path).build()
    assert isinstance(run, RunConfig)
    assert run.fit.batch_size == TrainConfig().batch_size
    assert run.sources == ["clean"]
