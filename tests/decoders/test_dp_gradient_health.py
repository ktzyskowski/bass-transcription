"""The whole DP must stay differentiable under degenerate and saturated evidence.

The companion to ``test_pitch_cumsums_gradients``, which pins one function. This drives
the real ``nll_chunk`` and asserts on the *gradient reaching the encoder logits*, which
is the only place the failure that cost two training runs was observable: a softmax bin
underflowing to 0.0 made ``sqrt``'s derivative infinite, and every forward value stayed
finite while the gradient became NaN. A test asserting on returned values could not
have caught it, and neither could any guard watching the loss.

The cases are the states a *trained* encoder reaches, which is the point -- an
untrained one is diffuse and safe, so a suite that only ever runs at initialisation
tests the regime where the bug does not exist. Hence a confidence sweep, saturated
binary heads, and a silent track.

The scorer's weights are randomised rather than left at init: several span-feature
weights start at zero, which zeroes the incoming gradient and would hide anything
downstream of them behind a clean-looking pass.
"""

import numpy as np
import pytest
import torch

from stemify.models.frames import frames_from_logits
from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.window import initial_window
from stemify.core.features import N_BINS
from stemify.music.notes import NoteEvent

DT, T = 0.0116, 240
TIMES = np.arange(T) * DT
NOTES = [
    NoteEvent(onset=0.2, offset=0.9, midi_pitch=45.0),
    NoteEvent(onset=1.2, offset=1.9, midi_pitch=50.0),
]


def _backward(
    pitch_scale: float | None = None, notes: bool = True, saturate: bool = False
) -> dict[str, torch.Tensor]:
    """Run one chunk's NLL and return the gradients at the encoder's logits."""
    torch.manual_seed(0)
    onset = torch.full((T,), 40.0 if saturate else 0.5).requires_grad_(True)
    voicing = torch.full((T,), -40.0 if saturate else 0.5).requires_grad_(True)
    pitch = (
        (torch.randn(T, N_BINS) * pitch_scale).requires_grad_(True)
        if pitch_scale is not None else None
    )
    cues = frames_from_logits(TIMES, onset=onset, voicing=voicing, pitch=pitch)

    decoder = SemiCRFDecoder(stride=3)
    with torch.no_grad():
        for p in decoder.scorer.parameters():
            p.normal_(0, 0.5)

    _, grid = decoder._prepare([cues])
    gold = gold_segments(TIMES[:: decoder.stride], NOTES if notes else [], grid.dmax)
    rows = decoder.scorer.labels.n_rows
    carry = initial_window(torch.zeros(grid.dmax, rows), 1, grid.dmax, rows)
    A, gold_score, _ = decoder.nll_chunk([cues], [gold], carry, 0, 0)

    loss = (A - gold_score).mean()
    assert torch.isfinite(loss), "forward is non-finite; this suite is about gradients"
    loss.backward()

    grads = {"onset": onset.grad, "voicing": voicing.grad}
    if pitch is not None:
        grads["pitch"] = pitch.grad
    return {k: v for k, v in grads.items() if v is not None}


@pytest.mark.parametrize("scale", [1, 10, 40, 200])
def test_the_dp_gradient_survives_a_confident_pitch_head(scale):
    """The regression. At scale >= ~20 the softmax underflows and the unclamped
    `sqrt` produced NaN here, with the loss still reporting a clean finite number."""
    for name, g in _backward(pitch_scale=scale).items():
        assert torch.isfinite(g).all(), f"non-finite {name} gradient at scale {scale}"


def test_saturated_binary_heads_do_not_produce_nan():
    """Onset and voicing pinned hard at both rails. The log-domain guards make this
    a *dead* gradient rather than a NaN one -- see the dead-zone test below, which
    pins that distinction so neither behaviour can change unnoticed."""
    for name, g in _backward(saturate=True, pitch_scale=40).items():
        assert torch.isfinite(g).all(), f"non-finite {name} gradient when saturated"


def test_a_silent_track_is_differentiable():
    """No notes at all: gold is all rest. Reachable -- plenty of Slakh chunks have no
    bass in them, and it is the shape of degenerate input that hides 0/0."""
    for name, g in _backward(notes=False, pitch_scale=40).items():
        assert torch.isfinite(g).all(), f"non-finite {name} gradient on a silent track"


def test_saturated_binary_heads_have_no_gradient_at_all():
    """Not a bug, but worth pinning: the clamps in `voicing_cumsums`/`onset_logit`
    that keep the forward finite also zero the derivative once a head saturates, so
    on those frames the CRF stops teaching onset and voicing entirely. Defensible --
    a frame the model is 1e-6 sure of has little left to learn -- but it is the same
    shape as the bug above (a guard that fixes the value and changes the gradient),
    and if it ever stops being true that should be a decision, not a surprise."""
    grads = _backward(saturate=True)
    assert float(grads["onset"].abs().max()) == 0.0
    assert float(grads["voicing"].abs().max()) == 0.0
    # ...while an unsaturated head does get one, so this is about saturation and not
    # about the gradient never arriving in the first place.
    mild = _backward()
    assert float(mild["onset"].abs().max()) > 0.0
    assert float(mild["voicing"].abs().max()) > 0.0
