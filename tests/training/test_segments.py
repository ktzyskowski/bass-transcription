"""Bounded carry chains: where a segment starts, and what the loop does at its end.

``segment_chunks`` cuts a track into independent chains so the alpha carry can be passed
*undetached* -- the only arrangement in which ``d(chunk k)/d(theta)`` reaches back
through ``alpha_{k-1}``, and so the only one whose step is a derivative of the loss it
reports (see ``test_training_dynamics``). Two things have to hold for that to be worth
anything, and neither is checked anywhere else.

**A chain must start where gold does.** A reset row's ``log Z`` sums only the tilings
reachable from its seed, while ``score(gold)`` is computed straight off the gold list
and consults no alpha at all. Seed at the chunk's edge and any gold segment reaching
back past it -- the normal case, since a segment belongs to the chunk holding its *end*
-- is scored but never normalised. ``log Z < score(gold)`` follows and the NLL, which is
a surprisal and cannot be negative, goes negative and unbounded.

**The loop must backward once per chain.** ``Trainer`` backwards each micro-batch, which
frees a graph the next chunk still holds a reference into; the coupling is declared by
``Loss.couples_micro_batches`` and the loop defers instead. Without that this raises
before any of the above matters.
"""

import numpy as np
import pytest
import torch

from stemify.core.conditions import CONDITIONS
from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.fit.dataset import ChunkDataset
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.semicrf.fit.tracks import train_items
from stemify.training.config import TrainConfig
from stemify.training.trainer import Trainer

# `track` is imported to be *injected*, so every test taking it as a parameter reads to
# ruff as a redefinition -- the standard false positive for a shared fixture. A
# `conftest.py` is the real fix; it means moving fixtures out of passing files.
from test_chunk_carry import (  # noqa: F401  (fixtures)
    CHUNK, DMAX, STRIDE, _NoOpCache, _Pointwise, track,
)
from test_chunked import _FakeDataset

SEGMENT = 2          # chunks per chain; the fixture's track is six, so three chains
CPU = torch.device("cpu")


def _dataset(items, cache, segment_chunks: int) -> ChunkDataset:
    return ChunkDataset(
        train_items(items, CONDITIONS["clean"], cache),
        chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE, batch_size=1,
        right_context=0, segment_chunks=segment_chunks,
    )


def _chunked(tmp_path, lengths, *, batch_size, segment_chunks):
    """A multi-track dataset, so streams roll over onto tracks of other lengths.

    Its own cache rather than the shared ``cache`` fixture: importing that one by
    name shadows the local ``cache`` this module unpacks from ``track`` everywhere
    else, and renaming those to dodge a lint is a worse trade than four lines.
    """
    store = _NoOpCache(tmp_path / "multi")
    ds = _FakeDataset(tmp_path, store, lengths)
    return ChunkDataset(
        train_items(ds, CONDITIONS["clean"], store),
        chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE, batch_size=batch_size,
        right_context=0, segment_chunks=segment_chunks,
    )


def test_a_segment_boundary_resets_every_stream_on_the_same_batch(track):  # noqa: F811
    """Synchronised on the *batch* index, which is the number the trainer counts.

    One backward covers a whole accumulation window, so a chain that straddles the
    boundary has its second half reach into a graph the first half's backward already
    freed. That makes agreement between the two counters a correctness condition, not a
    tidiness one -- and the counter has to be the batch, because a stream's own chunk
    index restarts at every track boundary and tracks are not all the same length.

    Caught in production, not here: keying on ``chunk_no`` passed the single-track
    fixture (where the two counters coincide) and crashed 20 minutes into a real run.
    """
    items, cache, _ = track
    resets = [b.resets[0] for b in _dataset(items, cache, SEGMENT)]
    assert resets == [i % SEGMENT == 0 for i in range(len(resets))]
    assert [b.resets[0] for b in _dataset(items, cache, 0)] == [
        True, *[False] * (len(resets) - 1)
    ], "segment_chunks=0 must still be one chain per track"


def test_streams_stay_in_lockstep_across_tracks_of_different_lengths(tmp_path):
    """The case the single-track fixture cannot express, and the one that crashed.

    Several tracks per stream, of *different* chunk counts, over several streams. A
    stream that rolls onto a new track restarts its own chunk index, so any rule keyed
    on that index drifts -- rows stop resetting together within an epoch, and the batch
    stops agreeing with the trainer's window. Both are silent right up to the backward.

    Lengths chosen so no track is a whole number of segments: that is what makes the
    two counters disagree, and equal-length tracks would hide it exactly as one track
    did.
    """
    ds = _chunked(tmp_path, [900, 500, 1300, 700], batch_size=2,
                  segment_chunks=SEGMENT)
    batches = list(ds)
    assert len(batches) > SEGMENT * 2, "need several windows to see any drift"
    for i, b in enumerate(batches):
        if i % SEGMENT == 0:
            # The only condition that has to hold. Every row starts a fresh chain where
            # the trainer will backward, so no carry ever crosses one.
            assert all(b.resets), (
                f"batch {i} reset rows {b.resets}, but the trainer steps every "
                f"{SEGMENT} batches -- a row that does not reset here carries a graph "
                "across the backward and reads it after it has been freed"
            )
    # Mid-window resets are *expected* and harmless: a stream rolling onto a new track
    # takes a fresh window, so it reaches back to nothing. It only makes that row's
    # chain shorter than the others', which costs a little exactness and no
    # correctness. Asserted present so this fixture is known to exercise the case.
    assert any(
        any(b.resets) for i, b in enumerate(batches) if i % SEGMENT != 0
    ), "no stream rolled onto a new track mid-window; the case is not being covered"


def test_a_chain_starts_where_its_gold_does_not_where_the_chunk_does(track):  # noqa: F811
    """The seed, read straight out of the window ``_fresh`` builds.

    ``initial_window``'s slot arithmetic is the scan's own: its first step is
    ``j = context + 1`` and reads window index ``k`` as position ``j - dmax + k``, so
    slot ``k`` is position ``context + 1 - dmax + k``. Inverting that is the whole
    claim, and an off-by-one here is silent -- every shape still matches and the NLL
    still falls, it is just normalising over a different set of tilings than it scores.
    """
    items, cache, _ = track
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE), detach_carry=False)
    batches = [b for b in _dataset(items, cache, SEGMENT) if b.resets[0]]
    assert len(batches) > 1, "the fixture must have a mid-track segment boundary"
    for batch in batches:
        window = loss._fresh(batch, CPU, DMAX)                  # (1, dmax, n_rows)
        seeded = (window[0] != window.min()).nonzero()
        assert seeded.shape[0] == 1, "exactly one reachable (position, label)"
        slot, label = int(seeded[0, 0]), int(seeded[0, 1])
        position = batch.context + 1 - DMAX + slot
        assert position == batch.golds[0][0][1], (
            "the chain is seeded at the chunk's edge rather than where its earliest "
            "gold segment begins, so that segment is scored but not normalised"
        )
        assert label == batch.prevs[0]


def test_the_nll_of_a_segmented_pass_is_a_surprisal(track):  # noqa: F811
    """``log Z >= score(gold)`` on every chunk, which is what makes it one.

    Gold is one of the tilings the partition sums, so its score cannot exceed the
    total -- unless it is not among them, which is exactly what seeding a chain at the
    wrong position does. Run with a perturbed scorer so the potentials are large enough
    for the omission to show: at the zero-ish init of an untrained scorer every
    potential is small and a missing term moves ``log Z`` by too little to cross zero.
    """
    items, cache, _ = track
    decoder = SemiCRFDecoder(stride=STRIDE)
    torch.manual_seed(3)
    with torch.no_grad():
        for p in decoder.parameters():
            p.add_(torch.randn_like(p) * 2.0)
    loss = SemiCRFLoss(decoder, detach_carry=False)
    model = _Pointwise()
    with torch.no_grad():
        nlls = [
            loss.compute(model, b, CPU)[1]["nll"]
            for b in _dataset(items, cache, SEGMENT)
        ]
    assert np.isfinite(nlls).all(), f"non-finite nll: {nlls}"
    assert min(nlls) >= 0.0, (
        f"nll {min(nlls)} is negative, so score(gold) exceeded log Z: gold's tiling is "
        f"not among the ones the partition sums. All: {nlls}"
    )


def test_the_loop_trains_a_whole_epoch_of_undetached_segments(track):  # noqa: F811
    """End to end, because the failure is a crash rather than a wrong number.

    ``Trainer`` backwards each micro-batch. With the carry undetached that frees a graph
    the next chunk reads, and torch raises "Trying to backward through the graph a
    second time" -- which is why ``detach_carry=False`` was unreachable through the
    production loop for as long as it existed, despite being documented as its point.
    """
    items, cache, _ = track
    model = _Pointwise()
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE), detach_carry=False)
    before = [p.detach().clone() for p in loss.parameters()]
    trainer = Trainer(model, loss, TrainConfig(
        learning_rate=1e-2, max_grad_norm=None, device="cpu",
        accumulate_steps=SEGMENT,
    ))
    metrics = trainer._run_epoch(_dataset(items, cache, SEGMENT), train=True)
    assert np.isfinite(metrics["total"])
    # Read from the weights, not from `.grad`: the loop zeroes those as part of the
    # step it just took, so a test that inspects them afterwards sees nothing and
    # passes for the wrong reason.
    assert any(
        not torch.allclose(a, b) for a, b in zip(before, loss.parameters(), strict=True)
    ), "an epoch ran without moving a single potential"


def test_the_loop_survives_streams_rolling_onto_new_tracks(tmp_path):
    """The run that crashed, at test scale: many tracks, many streams, one epoch.

    ``test_the_loop_trains_a_whole_epoch_of_undetached_segments`` is one track on one
    stream, where a chunk index and a batch index are the same number -- so it passed
    while the reset rule was keyed on the wrong one, and the real run died twenty
    minutes in with "Trying to backward through the graph a second time". Tracks of
    unequal length over two streams is the smallest fixture that can tell them apart.
    """
    model = _Pointwise()
    loss = SemiCRFLoss(SemiCRFDecoder(stride=STRIDE), detach_carry=False)
    trainer = Trainer(model, loss, TrainConfig(
        learning_rate=1e-3, max_grad_norm=None, device="cpu",
        accumulate_steps=SEGMENT,
    ))
    dataset = _chunked(tmp_path, [900, 500, 1300, 700], batch_size=2,
                       segment_chunks=SEGMENT)
    metrics = trainer._run_epoch(dataset, train=True)
    assert np.isfinite(metrics["total"])


def _segment_gradient(items, cache, *, detach_carry: bool) -> list[torch.Tensor]:
    """One chain's gradient, taken by hand so no optimiser step can clear it."""
    torch.manual_seed(11)
    model = _Pointwise()
    decoder = SemiCRFDecoder(stride=STRIDE)
    loss = SemiCRFLoss(decoder, detach_carry=detach_carry)
    params = list(decoder.parameters())
    total = torch.zeros(())
    for batch in list(_dataset(items, cache, SEGMENT))[:SEGMENT]:
        total = total + loss.compute(model, batch, CPU)[0]
    total.backward()
    return [
        (p.grad if p.grad is not None else torch.zeros_like(p)).clone() for p in params
    ]


def test_the_gradient_reaches_back_through_the_carry(track):  # noqa: F811
    """The whole point of an undetached chain, stated as a difference.

    Same chain, same weights, same loss value -- only whether the graph crosses the
    carry. If the two agreed, detaching would be free and there would be nothing to fix;
    the measured cost is that they do not, and by enough to flip the descent direction
    once the potentials sharpen (``test_training_dynamics``).
    """
    items, cache, _ = track
    truncated = _segment_gradient(items, cache, detach_carry=True)
    exact = _segment_gradient(items, cache, detach_carry=False)
    assert any(
        not torch.allclose(a, b, atol=1e-9) for a, b in zip(truncated, exact, strict=True)
    ), (
        "detaching the carry changed no gradient, so either the chains are one chunk "
        "long or the undetached window is not reaching the graph it should"
    )


def test_a_segment_length_the_step_cannot_land_on_is_refused():
    """``accumulate_steps`` *is* the chain length, and a mismatch is a crash minutes in.

    Earlier and the next chunk reads a freed graph; later and one step spans a reset,
    which silently mixes two chains' gradients. Neither is something to discover from a
    traceback partway through an epoch.
    """
    from stemify.semicrf.fit.recipe import SemiCRFRecipe

    recipe = SemiCRFRecipe(["clean"], segment_chunks=4)
    with pytest.raises(ValueError, match="accumulate_steps"):
        recipe._chunks("", "train", [CONDITIONS["clean"]], _NoOpCache("/tmp/nope"),
                       TrainConfig(accumulate_steps=2), None)
