"""Accumulation exists to hold the weights still, so that is what these assert.

The motive is not memory. A recurrent objective carries state between micro-batches --
the semi-CRF's alpha window -- and that state is produced by whatever the weights were
when it was computed. Move the weights every step and every chunk consumes a state its
own model never made: representational drift, and here the carry resets only once per
track (a median of 53 chunks), so the state is ~53 updates old by a track's end.

Holding the weights still for N micro-batches makes the carry exact within the window.
So the load-bearing property is **"the parameters do not change during a window"**, and
`test_weights_are_frozen_within_a_window` is the test that would fail if someone
"optimised" this back into a per-batch step. Everything else here protects the
arithmetic around it: that the accumulated gradient is a mean and not a sum (or the
clip threshold and the learning rate would both silently rescale with the dial), and
that no micro-batch is quietly dropped.
"""

import pytest
import torch
from torch import nn

from stemify.fitting.base import GRAD_NORM, Loss
from stemify.fitting.config import TrainConfig
from stemify.training.trainer import Trainer


class _Batch(list[int]):
    """The loop needs only ``__len__`` of a batch."""


class _LinearLoss(Loss[_Batch]):
    """``w * c`` for a scripted constant, so the gradient of each micro-batch is
    exactly that constant and the accumulated gradient has a known closed form."""

    def __init__(self, values: list[float]) -> None:
        self.w = nn.Parameter(torch.tensor(0.0))
        self._values = list(values)
        self.seen = 0

    def compute(self, model, batch, device):
        value = self._values[self.seen % len(self._values)]
        self.seen += 1
        loss = self.w * value
        return loss, {"total": float(loss.item())}

    def parameters(self):
        return [self.w]

    def state_dict(self):
        return {"w": self.w.detach().cpu().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.w.copy_(state["w"])


def _trainer(loss, accumulate, *, lr=0.1, clip=None, epochs=1):
    return Trainer(
        nn.Linear(1, 1), loss,
        TrainConfig(device="cpu", max_epochs=epochs, learning_rate=lr,
                    accumulate_steps=accumulate, max_grad_norm=clip),
    )


def _batches(n):
    return [_Batch([0]) for _ in range(n)]


# --------------------------------------------------------------------------- #
# The property accumulation exists for                                         #
# --------------------------------------------------------------------------- #
def test_weights_are_frozen_within_a_window():
    """The whole point. If the parameters move mid-window, the carry a later chunk
    reads was made by a model that no longer exists, and accumulation has bought
    nothing."""
    loss = _LinearLoss([1.0])
    trainer = _trainer(loss, accumulate=4)
    seen: list[float] = []
    original = loss.compute

    def spy(model, batch, device):
        seen.append(float(loss.w.item()))
        return original(model, batch, device)

    loss.compute = spy  # type: ignore[method-assign]
    trainer.fit(_batches(8))

    # Two windows of four. Within each, w must be identical; between them it moves.
    assert seen[0] == seen[1] == seen[2] == seen[3], "weights moved inside a window"
    assert seen[4] == seen[5] == seen[6] == seen[7], "weights moved inside a window"
    assert seen[0] != seen[4], "weights never moved at all; nothing was optimised"


def test_without_accumulation_the_weights_move_every_batch():
    """The contrast, so the test above cannot pass by the optimiser being broken."""
    loss = _LinearLoss([1.0])
    trainer = _trainer(loss, accumulate=1)
    seen: list[float] = []
    original = loss.compute

    def spy(model, batch, device):
        seen.append(float(loss.w.item()))
        return original(model, batch, device)

    loss.compute = spy  # type: ignore[method-assign]
    trainer.fit(_batches(4))
    assert len({round(v, 9) for v in seen}) == 4, "weights should move every batch"


# --------------------------------------------------------------------------- #
# The arithmetic around it                                                     #
# --------------------------------------------------------------------------- #
def test_one_optimiser_step_per_window():
    steps = []
    loss = _LinearLoss([1.0])
    trainer = _trainer(loss, accumulate=4)
    real = trainer.optimizer.step

    def counted(*a, **k):
        steps.append(1)
        return real(*a, **k)

    trainer.optimizer.step = counted  # type: ignore[method-assign]
    trainer.fit(_batches(12))
    assert len(steps) == 3


def test_the_accumulated_gradient_is_the_mean_not_the_sum():
    """Without the 1/N scaling the gradient grows with the dial, which would silently
    rescale both the clip threshold and the effective learning rate."""
    # d(w*c)/dw = c, so the window's mean gradient is mean([1,2,3,4]) = 2.5.
    # A summing implementation would report 10.0 here.
    loss = _LinearLoss([1.0, 2.0, 3.0, 4.0])
    epoch = _trainer(loss, accumulate=4)._run_epoch(_batches(4), train=True)
    assert epoch[GRAD_NORM] == pytest.approx(2.5, rel=1e-5)
    # The accumulator is reset after the step, so the next window starts clean.
    # (`zero_grad` sets grads to None rather than zeroing them in place.)
    assert loss.w.grad is None or float(loss.w.grad.abs().max()) == 0.0


def test_accumulate_one_matches_the_unaccumulated_loop():
    """The default must be a no-op, or every result recorded before this existed
    becomes incomparable."""
    a = _LinearLoss([1.0, 2.0, 3.0, 4.0])
    _trainer(a, accumulate=1).fit(_batches(4))
    b = _LinearLoss([1.0, 2.0, 3.0, 4.0])
    _trainer(b, accumulate=1).fit(_batches(4))
    assert a.w.item() == pytest.approx(b.w.item())
    # ...and it is genuinely stepping four times, not once.
    c = _LinearLoss([1.0, 2.0, 3.0, 4.0])
    _trainer(c, accumulate=4).fit(_batches(4))
    assert a.w.item() != pytest.approx(c.w.item())


def test_a_partial_final_window_still_steps():
    """5 batches at N=4 is one full window and a remainder. Dropping the remainder
    would silently discard up to N-1 micro-batches every epoch."""
    steps = []
    loss = _LinearLoss([1.0])
    trainer = _trainer(loss, accumulate=4)
    real = trainer.optimizer.step
    trainer.optimizer.step = lambda *a, **k: (steps.append(1), real(*a, **k))[1]  # type: ignore[method-assign]
    trainer.fit(_batches(5))
    assert len(steps) == 2, "the short final window was dropped"


def test_the_grad_norm_epoch_mean_is_not_diluted_by_the_dial():
    """GRAD_NORM exists on one micro-batch in N. Averaging it over *all* batches would
    under-report it by exactly N -- and it is the number the clip is tuned from."""
    loss = _LinearLoss([3.0])
    epoch = _trainer(loss, accumulate=4)._run_epoch(_batches(8), train=True)
    assert epoch[GRAD_NORM] == pytest.approx(3.0, rel=1e-5)


def test_clipping_applies_to_the_window_not_each_micro_batch():
    """Clipping per micro-batch would bound the sum of N clipped vectors rather than
    the gradient actually taken, so the clip would mean something different at every
    setting of the dial."""
    loss = _LinearLoss([100.0])          # mean window gradient 100, clip at 5
    trainer = _trainer(loss, accumulate=4, clip=5.0)

    # Assert on the gradient the optimiser was handed, not on the weight delta: the
    # optimiser is Adam, whose first step is ~lr whatever the magnitude, so a weight
    # delta would pass just as happily with no clipping at all.
    handed: list[float] = []
    real = trainer.optimizer.step

    def spy(*a, **k):
        handed.append(float(loss.w.grad.abs().max()))
        return real(*a, **k)

    trainer.optimizer.step = spy  # type: ignore[method-assign]
    epoch = trainer._run_epoch(_batches(4), train=True)

    assert epoch[GRAD_NORM] == pytest.approx(100.0, rel=1e-5)  # reported pre-clip
    assert handed == [pytest.approx(5.0, rel=1e-4)]            # applied post-clip

