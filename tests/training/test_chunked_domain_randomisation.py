"""``ChunkDataset`` over several input conditions: batches must actually mix them.

The structured objective used to refuse more than one condition, so a semi-CRF run
could not be domain-randomised while the frame path always was -- which made any
comparison between the two vary the objective *and* the condition set.

Lifting the restriction is not the whole job. Streams take ``items[b::batch_size]`` and
advance in near lockstep, so the *order* of the item list decides what a batch holds,
and two natural orderings both train on the union while never mixing conditions inside
a batch:

  - concatenated by condition -- every stream sits in the first block at the same time,
    so batches come out condition-homogeneous;
  - interleaved -- stream ``b`` only ever draws items of one parity, so each stream is
    permanently one condition.

Either would report the right ``--sources`` and still feed the TCN's BatchNorm one
condition's statistics at a time. So the shuffle is the feature, and this is its test.
"""

from pathlib import Path

import numpy as np
import pytest

from stemify.music.notes import NoteEvent
from stemify.persistence.serde import CachedItem
from stemify.semicrf.fit.dataset import ChunkDataset
from stemify.semicrf.fit.tracks import TrainItem

N_BINS, HOP_S = 12, 0.0116
STRIDE, CHUNK, DMAX = 3, 40, 30
N_CQT = 600
# Each track's features are a constant equal to its group's marker, so a batch's
# features say which conditions its rows came from. Real conditions differ by the audio
# they resolve to; this stands in for that without needing two transforms on disk.
CLEAN, SEPARATED = 1.0, 2.0


def _notes() -> list[NoteEvent]:
    span = N_CQT * HOP_S
    return [
        NoteEvent(onset=t, offset=t + 0.3, midi_pitch=40.0)
        for t in np.arange(0.05, span - 0.5, 0.4)
    ]


def _item(tmp: Path, name: str, marker: float) -> TrainItem:
    """One cached track whose every feature value is ``marker``."""
    path = tmp / f"{name}.npz"
    CachedItem(
        name,
        np.full((N_BINS, N_CQT), marker, dtype=np.float32),
        {},
        np.arange(N_CQT) * HOP_S,
    ).save(path)
    return TrainItem(features=(path,), notes=_notes())


@pytest.fixture
def items(tmp_path: Path) -> list[TrainItem]:
    """Sixteen tracks: the first eight 'clean', the last eight 'separated'.

    Grouped rather than interleaved on purpose -- this is exactly the order
    ``SemiCRFRecipe._chunks`` produces when it concatenates per-condition item lists,
    so it is the order the shuffle has to survive.
    """
    return (
        [_item(tmp_path, f"clean{k}", CLEAN) for k in range(8)]
        + [_item(tmp_path, f"sep{k}", SEPARATED) for k in range(8)]
    )


def _dataset(items, batch_size: int, seed: int = 0) -> ChunkDataset:
    return ChunkDataset(
        items, chunk_frames=CHUNK, dmax=DMAX, stride=STRIDE,
        batch_size=batch_size, seed=seed,
    )


def _markers(batch) -> set[float]:
    """Which condition each row of a batch came from.

    ``max``, not ``flatten()[0]``: a window is zero-padded on the left at a track's
    first chunk, so its leading value is 0.0 rather than the marker. Both markers are
    positive, so the maximum is the track's own value on every chunk.
    """
    return {float(batch.features[b].max()) for b in range(len(batch))}


def test_batches_mix_conditions(items):
    # The property domain randomisation actually needs, and the one both natural
    # orderings silently fail: a *single* batch must span conditions, so the TCN's
    # BatchNorm sees a blend rather than one condition's statistics per step.
    batches = list(_dataset(items, batch_size=4))
    assert batches, "the fixture should yield at least one batch"
    mixed = [b for b in batches if len(_markers(b)) > 1]
    assert mixed, (
        "no batch contained rows from both conditions -- the item list reached the "
        "streams in grouped order, so every stream was in the same block at once"
    )


def test_both_conditions_are_actually_reached(items):
    seen: set[float] = set()
    for batch in _dataset(items, batch_size=4):
        seen |= _markers(batch)
    assert seen == {CLEAN, SEPARATED}


def test_grouped_input_order_would_not_mix_without_the_shuffle(items):
    """The shuffle is load-bearing, not incidental.

    Reaching past the constructor to restore the grouped order reproduces what the
    unshuffled implementation did: streams take ``items[b::batch_size]`` from a list
    whose first half is one condition, so they all sit in that half together and no
    batch spans both. If this ever stops holding, the shuffle has stopped being the
    thing that makes the mixing work and this file is testing a coincidence.
    """
    dataset = _dataset(items, batch_size=4)
    dataset._items = list(items)          # undo the shuffle, keep everything else
    assert all(len(_markers(b)) == 1 for b in dataset)


def test_the_shuffle_is_seeded(items):
    """Reproducible across runs, and actually varied by the seed.

    Compared on the item *order* rather than on what the batches contain: with this
    fixture the mixing is thorough enough that two seeds give the same set of conditions
    in every batch, so a batch-level comparison would pass whether or not the seed did
    anything.
    """
    def order(seed: int) -> list[tuple[Path, ...]]:
        return [i.features for i in _dataset(items, batch_size=4, seed=seed)._items]

    assert order(7) == order(7)
    assert order(1) != order(2)


def test_a_single_condition_still_works(tmp_path):
    # The common case must not have regressed: one condition, no mixing to do.
    single = [_item(tmp_path, f"only{k}", CLEAN) for k in range(8)]
    batches = list(_dataset(single, batch_size=4))
    assert batches
    assert all(_markers(b) == {CLEAN} for b in batches)
