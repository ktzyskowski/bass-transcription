"""The semi-CRF's auxiliary pitch term: ``NLL + w * pitch_ce`` over a chunk's frames.

The structured NLL cannot supervise the pitch head's *values* -- the span features read
how far the distribution travels across a span and how much its frames disagree, and
both are invariant to which pitch it names. So the head needs a per-frame term, and
these cover the three things that can silently go wrong with one: it is not applied at
all, it is applied to the wrong frames, or it is applied to a model that has no head.

No audio and no real backbone: a per-frame linear tagger with a pitch branch, over
features that encode the answer.
"""

import numpy as np
import pytest
import torch
from torch import nn

from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY
from stemify.models.heads import pitch_ce
from stemify.music.notes import NoteEvent
from stemify.semicrf.fit.dataset import Chunk, collate
from stemify.semicrf.fit.loss import SemiCRFLoss

DT = 0.0116
T = 240
TIMES = np.arange(T) * DT
N_BINS = 4
STRIDE = 3
PITCH_BINS = 216
CPU = torch.device("cpu")


class _HeadsWithPitch(nn.Module):
    """onset/voicing per frame, plus a pitch branch over ``PITCH_BINS``."""

    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.onset = nn.Linear(n_bins, 1)
        self.voicing = nn.Linear(n_bins, 1)
        self.pitch = nn.Linear(n_bins, PITCH_BINS)

    def forward(self, x):  # (B, 1, n_bins, T)
        h = x.squeeze(1).transpose(1, 2)  # (B, T, n_bins)
        return {
            "onset": self.onset(h),      # (B, T, 1)
            "voicing": self.voicing(h),  # (B, T, 1)
            "pitch": self.pitch(h),      # (B, T, PITCH_BINS)
        }


class _HeadsNoPitch(nn.Module):
    def __init__(self, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.onset = nn.Linear(n_bins, 1)
        self.voicing = nn.Linear(n_bins, 1)

    def forward(self, x):
        h = x.squeeze(1).transpose(1, 2)
        return {"onset": self.onset(h), "voicing": self.voicing(h)}


def _chunk(spans, stream: int, *, midi: float | None = 45.0) -> Chunk:
    """One chunk whose features mark onsets and voiced frames.

    ``midi`` is written into the pitch target on voiced frames and left NaN elsewhere,
    matching what ``target_windows`` produces off a real track.
    """
    feats = np.zeros((N_BINS, T), dtype=np.float32)
    y_pitch = np.full(T, np.nan, dtype=np.float32)
    for a, b in spans:
        feats[0, round(a / DT)] = 1.0
        voiced = (TIMES >= a) & (TIMES < b)
        feats[1, voiced] = 1.0
        if midi is not None:
            y_pitch[voiced] = midi
    notes = [NoteEvent(onset=a, offset=b, midi_pitch=45.0) for a, b in spans]
    dmax = SemiCRFDecoder(stride=STRIDE).scorer.grid(TIMES[::STRIDE]).dmax
    return Chunk(
        features=feats[None],
        frame_times=TIMES,
        gold=gold_segments(TIMES[::STRIDE], notes, dmax),
        y={"pitch": y_pitch},
        stream=stream,
        reset=True,
        prev=BINARY.START,
    )


def _batch(context: int = 0, right: int = 0, **kw):
    return collate(
        [_chunk([(0.2, 0.9), (1.2, 1.9)], stream=0, **kw),
         _chunk([(0.3, 0.8), (1.4, 2.0)], stream=1, **kw)],
        context=context,
        right=right,
    )


def _loss(**kw) -> SemiCRFLoss:
    return SemiCRFLoss(SemiCRFDecoder(stride=STRIDE), **kw)


def test_without_aux_heads_the_total_is_just_the_nll():
    # The pre-existing behaviour has to survive: a run with no pitch head is unchanged.
    _, metrics = _loss().compute(_HeadsWithPitch(), _batch(), CPU)
    assert metrics["total"] == metrics["nll"]
    assert "pitch" not in metrics


def test_the_aux_term_is_added_and_reported_separately():
    total, metrics = _loss(aux_heads=["pitch"]).compute(
        _HeadsWithPitch(), _batch(), CPU
    )
    assert torch.isfinite(total) and total.requires_grad
    # total == nll + 1.0 * pitch, and the pitch term is real (a random head is wrong).
    assert metrics["pitch"] > 0.0
    assert metrics["total"] == pytest.approx(metrics["nll"] + metrics["pitch"], rel=1e-6)


def test_the_weight_scales_only_the_aux_term():
    model, batch = _HeadsWithPitch(), _batch()
    _, one = _loss(aux_heads=["pitch"], aux_weights={"pitch": 1.0}).compute(model, batch, CPU)
    _, three = _loss(aux_heads=["pitch"], aux_weights={"pitch": 3.0}).compute(model, batch, CPU)
    assert three["nll"] == pytest.approx(one["nll"], rel=1e-6)
    assert three["pitch"] == pytest.approx(one["pitch"], rel=1e-6)
    assert three["total"] == pytest.approx(one["nll"] + 3 * one["pitch"], rel=1e-6)


def test_the_gradient_reaches_the_pitch_head():
    # Without this the head trains on nothing -- the whole reason the term exists.
    model = _HeadsWithPitch()
    total, _ = _loss(aux_heads=["pitch"]).compute(model, _batch(), CPU)
    total.backward()
    assert model.pitch.weight.grad is not None
    assert torch.any(model.pitch.weight.grad != 0)


def test_only_the_chunks_own_frames_are_scored():
    """The context and right margins belong to the neighbouring chunks.

    They are scored again as part of *their* chunk, so including them here would train
    those frames twice per epoch at a weight set by where the boundaries happened to
    fall. Checked by giving the margins a target the model is confidently wrong about:
    if they were scored, the loss would move.
    """
    context, right = 8, 5           # DP frames
    lo, hi = context * STRIDE, T - right * STRIDE

    model = _HeadsWithPitch()
    batch = _batch(context=context, right=right)
    _, scored = _loss(aux_heads=["pitch"]).compute(model, batch, CPU)

    # Corrupt only the margins -- the frames the slice must exclude.
    poisoned = _batch(context=context, right=right)
    poisoned.y["pitch"][:, :lo] = 21.0
    poisoned.y["pitch"][:, hi:] = 108.0
    _, after = _loss(aux_heads=["pitch"]).compute(model, poisoned, CPU)

    assert after["pitch"] == pytest.approx(scored["pitch"], rel=1e-6)


def test_the_scored_slice_matches_pitch_ce_over_those_frames_exactly():
    # Pins the term to the shared implementation rather than to an internal recompute:
    # an off-by-one in the slice would pass every test above and fail this one.
    context, right = 8, 5
    lo, hi = context * STRIDE, T - right * STRIDE
    model = _HeadsWithPitch()
    batch = _batch(context=context, right=right)

    _, metrics = _loss(aux_heads=["pitch"]).compute(model, batch, CPU)
    with torch.no_grad():
        logits = model(batch.features)["pitch"]
        expected = pitch_ce(logits[:, lo:hi], batch.y["pitch"][:, lo:hi])
    assert metrics["pitch"] == pytest.approx(float(expected), rel=1e-6)


def test_nan_padding_past_a_tracks_end_contributes_nothing():
    # target_windows NaN-pads off the track; pitch_ce masks on it. A chunk that is
    # entirely unvoiced therefore has no pitch signal, and must still be finite.
    _, metrics = _loss(aux_heads=["pitch"]).compute(
        _HeadsWithPitch(), _batch(midi=None), CPU
    )
    assert metrics["pitch"] == 0.0
    assert np.isfinite(metrics["total"])


def test_a_model_without_a_pitch_head_is_refused_by_name():
    # Silently skipping the term is the failure this guards: the head would not exist,
    # the loss would look healthy, and the run would report a pitch model it never had.
    with pytest.raises(ValueError, match="no such branch"):
        _loss(aux_heads=["pitch"]).compute(_HeadsNoPitch(), _batch(), CPU)


def test_a_batch_without_the_target_is_refused_by_name():
    batch = _batch()
    batch.y.pop("pitch")
    with pytest.raises(KeyError, match="carries no such target"):
        _loss(aux_heads=["pitch"]).compute(_HeadsWithPitch(), batch, CPU)


def test_an_unknown_aux_head_is_refused_at_construction():
    # Not at the first batch, an epoch into a run.
    with pytest.raises(NotImplementedError, match="no auxiliary term"):
        _loss(aux_heads=["string"])
