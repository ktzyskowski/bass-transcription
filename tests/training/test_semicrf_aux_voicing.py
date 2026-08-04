"""The auxiliary voicing term, and why it exists.

The structured NLL supervises voicing only *competitively*. It is discriminative over
segmentations, so a feature earns gradient in proportion to how much it helps rank gold
above the alternatives, and ``LinearSpanScorer`` gives every feature a learned scalar.
Voicing is largely redundant given onset evidence, the duration prior and the transition
table -- its unique contribution is spotting rests, which are rare -- so ``w_v``/``w_u``
can shrink toward zero and leave the head with no signal at all.

That is not a hypothesis: the first stable semi-CRF run measured voicing AUROC **0.516**
-- chance -- where the same head under frame BCE reaches **0.984**. Pitch is the
stronger version of the same story (the span features are invariant to which pitch is
named, so the NLL cannot supervise its values at all), which is why that term came
first.

These pin the term's *mechanics*. Whether it lifts AUROC is an experiment, not a test.
"""

import numpy as np
import pytest
import torch
from torch import nn

from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.models.heads import frame_bce
from stemify.notes import NoteEvent
from stemify.serde import CachedItem
from stemify.training.recipes import RECIPES
from stemify.training.semicrf.dataset import ChunkDataset
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.semicrf.tracks import TrainItem
from stemify.features import N_BINS

HOP_S, N_CQT = 0.0116, 4000
TIMES = np.arange(N_CQT) * HOP_S


@pytest.fixture
def batch(tmp_path):
    items = []
    for k in range(2):
        notes = [NoteEvent(onset=0.5 + i * 0.7, offset=1.0 + i * 0.7, midi_pitch=40.0)
                 for i in range(40)]
        voiced = np.zeros(N_CQT, dtype=np.float32)
        midi = np.full(N_CQT, np.nan, dtype=np.float32)
        for n in notes:
            m = (TIMES >= n.onset) & (TIMES < n.offset)
            voiced[m], midi[m] = 1.0, n.midi_pitch
        path = tmp_path / f"t{k}.npz"
        CachedItem(f"t{k}", np.random.randn(N_BINS, N_CQT).astype(np.float32) * 0.5,
                   {"voicing": voiced, "pitch": midi}, TIMES).save(path)
        items.append(TrainItem(features=(path,), notes=notes))
    # right_context is non-zero on purpose: with context == right == 0 the scored
    # slice *is* the whole window, and a term that ignored the slice entirely would
    # pass every assertion below.
    ds = ChunkDataset(items, chunk_frames=128, dmax=30, stride=3, batch_size=2,
                      heads=("pitch", "voicing"), right_context=4)
    return next(iter(ds))


def _model():
    torch.manual_seed(0)
    return RECIPES["semicrf"](["clean"], pitch=True).build_model()


# --------------------------------------------------------------------------- #
# The term fires, and its weight is its own                                    #
# --------------------------------------------------------------------------- #
def test_the_voicing_term_is_reported_and_changes_the_total(batch):
    model, dev = _model(), torch.device("cpu")
    off = SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("pitch",))
    on = SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("pitch", "voicing"))
    _, m_off = off.compute(model, batch, dev)
    _, m_on = on.compute(model, batch, dev)
    assert "voicing" not in m_off
    assert m_on["voicing"] > 0.0
    assert m_on["total"] == pytest.approx(m_off["total"] + m_on["voicing"], rel=1e-5)


def test_each_head_carries_its_own_weight(batch):
    """One shared coefficient would mean whichever head was tuned last: pitch is a
    categorical CE over 216 bins on voiced frames, voicing is BCE over every frame."""
    model, dev = _model(), torch.device("cpu")
    loss = SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("pitch", "voicing"),
                       aux_weights={"pitch": 1.0, "voicing": 3.0})
    _, m = loss.compute(model, batch, dev)
    base = SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("pitch", "voicing"))
    _, b = base.compute(model, batch, dev)
    # The reported term is the unweighted loss; only the total scales.
    assert m["voicing"] == pytest.approx(b["voicing"], rel=1e-5)
    assert m["total"] == pytest.approx(b["total"] + 2.0 * b["voicing"], rel=1e-5)


@pytest.mark.parametrize("head", ["voicing", "onset"])
def test_a_zero_weight_drops_the_head_rather_than_scaling_it(head):
    """Weight 0 must remove the head from `aux_heads` entirely, not multiply its term
    by zero -- otherwise the dataset still caches and serves a target for a term that
    contributes nothing, and the run pays for a label it never reads."""
    off = RECIPES["semicrf"](["clean"], pitch=True, **{f"{head}_weight": 0.0})
    assert head not in off.aux_heads
    off.build_loss()          # the weights dict must not name it either
    assert head in RECIPES["semicrf"](["clean"], pitch=True).aux_heads


def test_both_binary_heads_are_supervised_by_default():
    """Measured: the structured NLL leaves onset AP at 0.20-0.53 and voicing AUROC at
    ~0.51, where frame BCE reaches 0.994 and 0.995 for the same heads through the same
    validator. Neither is adequately supervised by the segmentation alone."""
    assert RECIPES["semicrf"](["clean"], pitch=True).aux_heads == [
        "pitch", "onset", "voicing"
    ]


def test_a_weight_for_an_undeclared_head_is_refused():
    with pytest.raises(ValueError, match="not in aux_heads"):
        SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("pitch",),
                    aux_weights={"voicing": 1.0})


def test_an_unsupported_head_is_still_refused():
    with pytest.raises(NotImplementedError, match="no auxiliary term"):
        SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("tablature",))


# --------------------------------------------------------------------------- #
# The masking, which is why frame_bce exists at all                            #
# --------------------------------------------------------------------------- #
def test_nan_padding_past_a_tracks_end_does_not_poison_the_term():
    """The last chunk of a track runs off the end, where `target_windows` writes NaN.
    Unmasked, those frames make the whole term NaN -- a total failure of a loss that
    still looks healthy, which is exactly the class of bug this repo keeps paying for.
    """
    logits = torch.zeros(2, 8, requires_grad=True)
    target = torch.zeros(2, 8)
    target[:, 5:] = float("nan")          # the run-off
    loss = frame_bce(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # ...and the masked frames contributed nothing.
    assert float(logits.grad[:, 5:].abs().max()) == 0.0


def test_an_all_nan_target_stays_in_the_graph():
    """Zero, not absent: a batch with nothing to learn must still be differentiable, or
    it silently contributes no gradient rather than contributing none."""
    logits = torch.zeros(2, 4, requires_grad=True)
    loss = frame_bce(logits, torch.full((2, 4), float("nan")))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_the_term_scores_only_the_frames_the_nll_scores(batch):
    """Shared slice. A term averaging over the whole window would divide by a different
    frame count than the NLL, so the per-frame unit that makes the weights meaningful
    would stop holding -- and it would supervise the margins twice, since one chunk's
    context is the previous chunk's scored region."""
    model, dev = _model(), torch.device("cpu")
    loss = SemiCRFLoss(SemiCRFDecoder(stride=3), aux_heads=("voicing",))
    out = model(batch.features)
    lo, hi = loss.scored_slice(batch, out["voicing"].shape[1])
    expected = frame_bce(out["voicing"][:, lo:hi], batch.y["voicing"][:, lo:hi])
    _, metrics = loss.compute(model, batch, dev)
    assert metrics["voicing"] == pytest.approx(expected.item(), rel=1e-5)
    # Strictly inside the window, or the assertion above proves nothing.
    assert lo > 0 or batch.context == 0
    assert hi < out["voicing"].shape[1], "right margin not excluded; test has no teeth"
