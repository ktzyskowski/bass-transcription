"""Divergence must stop the run, not ride along inside it.

The instance these cover: a semi-CRF+pitch run went NaN in both terms at once and kept
training for another 2500 steps, because nothing in the loop distinguishes NaN from a
bad epoch. ``nan < best_val`` is False, so early stopping never fires and no checkpoint
is ever taken as best; the loop would have reached ``max_epochs`` and saved NaN weights
through the ordinary success path. The same family as the empty-loader guards -- a
degenerate value travelling silently -- except here the destination is a saved artifact
that looks like a trained model.

Two halves, and both are needed. The clip is what stops the ordinary cause (one
outsized gradient from a structured NLL over an unusual chunk); the finite guard is
what makes anything the clip cannot catch visible in minutes instead of an afternoon.
Testing only the guard would leave every recoverable run dying, and testing only the
clip would leave the unrecoverable ones silent.
"""

import math

import pytest
import torch
from torch import nn

from stemify.training.base import GRAD_NORM, Loss
from stemify.training.config import TrainConfig
from stemify.training.trainer import Trainer


class _Batch(list[int]):
    """A batch is only ``__len__`` to the loop, so an int list is a whole batch."""


class _ScriptedLoss(Loss[_Batch]):
    """Emits a scripted sequence of loss values through a real parameter.

    Real, because the guards sit around ``backward``/``step``: a detached constant
    would never populate ``.grad`` and the clip would have nothing to measure. The
    scripted value multiplies the parameter, so it reaches the gradient too.
    """

    def __init__(self, values: list[float]) -> None:
        self.w = nn.Parameter(torch.tensor(1.0))
        self._values = iter(values)
        self.seen = 0

    def compute(self, model, batch, device):
        self.seen += 1
        loss = self.w * next(self._values)
        return loss, {"total": float(loss.item())}

    def parameters(self):
        return [self.w]

    def state_dict(self):
        return {"w": self.w.detach().cpu().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.w.copy_(state["w"])


def _trainer(values, max_grad_norm=5.0, epochs=1):
    loss = _ScriptedLoss(values)
    trainer = Trainer(
        nn.Linear(1, 1),
        loss,
        TrainConfig(device="cpu", max_epochs=epochs, max_grad_norm=max_grad_norm),
    )
    return trainer, loss


def _batches(n: int) -> list[_Batch]:
    return [_Batch([0]) for _ in range(n)]


# --------------------------------------------------------------------------- #
# The finite guard: stop at the step that went bad                             #
# --------------------------------------------------------------------------- #
def test_a_nan_loss_stops_the_run():
    trainer, _ = _trainer([1.0, float("nan"), 1.0])
    with pytest.raises(ValueError, match="non-finite loss"):
        trainer.fit(_batches(3))


def test_an_infinite_loss_stops_the_run_too():
    # inf is the step before nan, not a separate condition: inf - inf is what makes
    # the nan. Catching only nan would let the run continue through the cause.
    trainer, _ = _trainer([1.0, float("inf")])
    with pytest.raises(ValueError, match="non-finite loss"):
        trainer.fit(_batches(2))


def test_the_run_stops_at_the_bad_step_not_at_the_end_of_the_epoch():
    """The whole point: 2500 further steps is what this costs when it does not hold."""
    trainer, loss = _trainer([1.0, 1.0, float("nan"), 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        trainer.fit(_batches(6))
    assert loss.seen == 3, "kept computing batches after the loss went non-finite"


def test_the_error_names_the_global_step():
    trainer, _ = _trainer([1.0, 1.0, float("nan")])
    with pytest.raises(ValueError, match="global step 3"):
        trainer.fit(_batches(3))


def test_the_step_is_named_even_without_a_batch_callback():
    """The counter used to advance only when a callback was listening, so an untracked
    run would have reported every divergence as step 1."""
    trainer, _ = _trainer([1.0, 1.0, 1.0, float("nan")])
    with pytest.raises(ValueError, match="global step 4"):
        trainer.fit(_batches(4))  # no callbacks -> no on_batch


def test_the_error_carries_the_remedy():
    trainer, _ = _trainer([float("nan")])
    with pytest.raises(ValueError, match="--learning-rate.*--max-grad-norm"):
        trainer.fit(_batches(1))


def test_a_nan_validation_loss_stops_the_run():
    # Distinct path: val runs under no_grad and never reaches the clip, so a NaN there
    # would otherwise flow straight into `nan < best_val` and silently never checkpoint.
    trainer, _ = _trainer([1.0, float("nan")])
    with pytest.raises(ValueError, match="during validation"):
        trainer.fit(_batches(1), _batches(1))


def test_a_finite_run_is_untouched():
    trainer, loss = _trainer([1.0] * 4, epochs=2)
    history = trainer.fit(_batches(2))
    assert loss.seen == 4
    assert all(math.isfinite(v) for v in history.train_loss)


# --------------------------------------------------------------------------- #
# The clip: the ordinary cause, caught before it reaches the weights           #
# --------------------------------------------------------------------------- #

def _applied_grad(trainer, loss, batches=1) -> float:
    """The gradient norm the optimiser was actually handed, i.e. post-clip.

    Captured at `step` rather than read off `.grad` afterwards, because the loop zeroes
    the accumulator once a step is applied. Distinct from the reported GRAD_NORM, which
    is deliberately the *pre*-clip norm.
    """
    real = trainer.optimizer.step
    seen: list[float] = []

    def spy(*a, **k):
        seen.append(float(loss.w.grad.norm()))
        return real(*a, **k)

    trainer.optimizer.step = spy
    trainer.fit(_batches(batches))
    assert seen, "the optimiser never stepped"
    return seen[0]


def test_an_outsized_gradient_is_clipped_to_the_limit():
    # d(w * 1000)/dw = 1000, an order of magnitude past any sane step.
    trainer, loss = _trainer([1000.0], max_grad_norm=5.0)
    # Adam's first step is ~lr regardless of magnitude, so assert on the gradient the
    # optimiser was handed, which is what clipping actually changes.
    assert _applied_grad(trainer, loss) == pytest.approx(5.0, rel=1e-4)


def test_an_ordinary_gradient_passes_through_untouched():
    # The clip is a safety net, not a regulariser: below the limit it must do nothing.
    trainer, loss = _trainer([2.0], max_grad_norm=5.0)
    assert _applied_grad(trainer, loss) == pytest.approx(2.0, rel=1e-4)


def test_clipping_can_be_disabled_and_then_does_not_clip():
    trainer, loss = _trainer([1000.0], max_grad_norm=None)
    assert _applied_grad(trainer, loss) == pytest.approx(1000.0, rel=1e-4)


def test_the_norm_is_still_measured_when_clipping_is_off():
    """Disabling the clip must not also blind the run -- the metric is how a threshold
    gets chosen from data rather than guessed."""
    trainer, _ = _trainer([1000.0], max_grad_norm=None)
    epoch = trainer._run_epoch(_batches(1), train=True)
    assert epoch[GRAD_NORM] == pytest.approx(1000.0, rel=1e-4)


def test_the_pre_clip_norm_is_reported_not_the_post_clip_one():
    # Post-clip the number is either itself or exactly max_grad_norm, which says
    # nothing about how far out the step was.
    trainer, _ = _trainer([1000.0], max_grad_norm=5.0)
    epoch = trainer._run_epoch(_batches(1), train=True)
    assert epoch[GRAD_NORM] == pytest.approx(1000.0, rel=1e-4)


def test_the_norm_reaches_the_batch_callback():
    trainer, _ = _trainer([3.0, 3.0])
    seen: list[dict[str, float]] = []

    class _Recorder:
        def on_batch_end(self, step, losses):
            seen.append(dict(losses))

        def on_epoch_end(self, logs): ...
        def on_train_end(self, best_epoch): ...

    trainer.fit(_batches(2), callbacks=[_Recorder()])
    assert seen and all(GRAD_NORM in m for m in seen)


def test_validation_epochs_carry_no_grad_norm():
    # Nothing to measure under no_grad, and a key present for train but not val would
    # otherwise read as "the norm went to zero at validation".
    trainer, _ = _trainer([1.0])
    assert GRAD_NORM not in trainer._run_epoch(_batches(1), train=False)


def test_a_non_finite_gradient_stops_the_run_before_the_step():
    """A finite loss can still backward into a non-finite gradient, and clipping cannot
    rescue one -- scaling inf leaves inf. The weights are still good at that point, so
    this is the last place the run can stop clean."""

    class _InfGrad(_ScriptedLoss):
        def __init__(self):
            super().__init__([])
            self.w = nn.Parameter(torch.tensor(0.0))

        def compute(self, model, batch, device):
            self.seen += 1
            # sqrt at 0: the value is 0.0 and perfectly finite, while d/dw is
            # 1/(2*sqrt(w)) = inf. So the loss guard has nothing to catch and the
            # gradient guard is the only thing between this and poisoned weights.
            loss = self.w.sqrt()
            return loss, {"total": float(loss.item())}

    loss = _InfGrad()
    trainer = Trainer(nn.Linear(1, 1), loss, TrainConfig(device="cpu", max_epochs=1))
    before = loss.w.detach().clone()
    with pytest.raises(ValueError, match="non-finite gradient"):
        trainer.fit(_batches(1))
    assert torch.equal(loss.w.detach(), before), "stepped on a non-finite gradient"
