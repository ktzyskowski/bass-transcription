"""The span features must stay differentiable once the pitch head gets *confident*.

The bug these cover killed two training runs. ``pitch_cumsums`` took ``p.sqrt()`` on a
softmax; ``d(sqrt p)/dp = 1/(2 sqrt p)`` is unbounded as ``p -> 0``, and a softmax over
216 bins underflows to exactly 0.0 in float32 once the logit spread passes ~104. The
backward pass then computes ``1/0 -> inf`` and softmax's own backward turns that into
``0 * inf -> NaN``, **while the forward value stays perfectly finite** -- so nothing
downstream of the loss could see it and every guard that watches the loss was blind.

Two properties are worth separating, because only one of them is about NaN:

1. The gradient is finite even when bins underflow. That is the crash.
2. The gradient stays *bounded* as the head sharpens. That is the damage before the
   crash, and it is the one that explains the training curves: gradients of order
   1e10 flowing into the shared trunk for thousands of steps, growing as the head
   grows confident, so the pitch branch is punished precisely for learning and is
   driven back toward uniform. Testing only (1) would let a fix that merely dodges
   the exact zero pass while the run still slowly dies.

Everything here is about the *gradient*, so an assertion on a returned value cannot
substitute -- the forward was always correct.
"""

import pytest
import torch

from stemify.decoders.semicrf.cumsums import pitch_cumsums
from stemify.features import N_BINS


def _grad(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.clone().requires_grad_(True)
    path_c, entropy_c, pitch_c = pitch_cumsums(logits)
    (path_c.sum() + entropy_c.sum() + pitch_c.sum()).backward()
    assert logits.grad is not None
    return logits.grad


def _logits(scale: float, frames: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(frames, N_BINS) * scale


@pytest.mark.parametrize("scale", [1, 10, 20, 40, 80, 200])
def test_gradients_are_finite_at_every_confidence(scale):
    assert torch.isfinite(_grad(_logits(scale))).all(), (
        f"non-finite gradient at logit scale {scale}"
    )


def test_the_regime_that_broke_it_really_does_underflow():
    """Pins the premise, so this suite cannot pass by the bins never reaching zero.

    Without it a change to N_BINS or to the frame count could quietly move every case
    above into the safe regime, and the tests would go on passing while testing
    nothing.
    """
    p = torch.softmax(_logits(40), dim=-1)
    assert (p == 0).any(), "no bin underflowed; this suite would prove nothing"


def test_the_forward_value_was_never_the_problem():
    """The reason this went unnoticed: every value is finite in the failing regime."""
    for out in pitch_cumsums(_logits(40)):
        assert torch.isfinite(out).all()


def test_gradients_do_not_blow_up_as_the_head_sharpens():
    """The damage before the crash. A confident head must not produce a gradient
    orders of magnitude past a diffuse one -- that is the feedback loop that drove
    `pitch` back to uniform over ~3000 steps in both diverged runs."""
    diffuse = _grad(_logits(1)).abs().max().item()
    confident = _grad(_logits(80)).abs().max().item()
    assert confident < 100 * diffuse, (
        f"gradient grew {confident / diffuse:.1f}x as the head sharpened "
        f"({diffuse:.3g} -> {confident:.3g}); a confident head is being punished"
    )


def test_the_clamp_does_not_move_the_bhattacharyya_coefficient():
    """The guard is for the derivative, not the value: on ordinary logits the path
    feature must be what it was, or this traded a crash for a silent change of
    objective."""
    logits = _logits(3, seed=1)
    p = torch.softmax(logits, dim=-1)
    unclamped = 1.0 - (p.sqrt()[1:] * p.sqrt()[:-1]).sum(-1)
    path_c = pitch_cumsums(logits)[0]
    # path_c is the prefix sum of [0, *path]; recover the per-frame terms.
    assert torch.allclose(path_c.diff()[1:], unclamped, atol=1e-5)
