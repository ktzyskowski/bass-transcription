"""ChunkedSequenceDataset: chunk geometry, stream ordering, and the property that
matters -- chunks fed through the carry reproduce the whole-track log Z.

No audio and no model: a fake Slakh-like dataset writes synthetic CQTs into a tmp
cache, so the geometry (padding, context, gold split) is exercised exactly as training
would drive it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from stemify.cache import Cache
from stemify.training.conditions import CONDITIONS
from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.decoders.semicrf.emissions import build_emissions, pad_emissions
from stemify.decoders.semicrf.forward import batched_log_partition, chunk_log_partition
from stemify.notes import NoteEvent
from stemify.serde import TrackExample
from stemify.sources import Sources
from stemify.training.datasets.chunked import ChunkedSequenceDataset, collate

N_BINS, HOP_S = 12, 0.0116
STRIDE = 3


@dataclass
class _Item:
    name: str
    n_cqt: int
    root: Path

    def audio(self) -> str:
        return str(self.root / f"{self.name}.wav")

    def sources(self) -> Sources:
        # Mirrors EvalItem's default. These fakes have only the one rendering, so
        # every channel of a condition resolves to the same file -- enough to
        # exercise the geometry, which is what this module tests.
        return Sources(bass=self.audio(), mix=self.audio())

    def reference_notes(self) -> list[NoteEvent]:
        # back-to-back ~0.4 s notes, roughly bass-line density
        span = self.n_cqt * HOP_S
        return [
            NoteEvent(onset=t, offset=t + 0.3, midi_pitch=40.0)
            for t in np.arange(0.05, span - 0.5, 0.4)
        ]


class _FakeDataset(list):
    """A Slakh-like dataset of synthetic tracks, with their CQTs already cached."""

    def __init__(self, tmp: Path, cache: Cache, lengths: list[int]) -> None:
        super().__init__()
        rng = np.random.default_rng(0)
        for k, n in enumerate(lengths):
            item = _Item(f"t{k}", n, tmp)
            Path(item.audio()).write_bytes(b"")
            feats = rng.standard_normal((N_BINS, n)).astype(np.float32)
            TrackExample(
                item.name, feats, {}, np.arange(n) * HOP_S
            ).save(cache.feature_path(item.audio()))
            self.append(item)


class _NoOpCache(Cache):
    """`ensure_features` is a no-op: the fixtures write their CQTs directly, and
    `contour` returns a synthetic pyin track rather than running pyin on empty audio."""

    def ensure_features(self, dataset, transform=None, role="bass"):
        return [self.feature_path(i.sources().role(role)) for i in dataset]

    def contour(self, estimator, audio):
        n = 400
        return Contour(times=np.arange(n) * HOP_S, freqs=np.full(n, 110.0))


@pytest.fixture
def cache(tmp_path):
    return _NoOpCache(tmp_path / "cache")


def _dataset(tmp_path, cache, lengths, chunk, dmax, batch_size, right=0):
    ds = _FakeDataset(tmp_path, cache, lengths)
    return ChunkedSequenceDataset(
        ds, cache, CONDITIONS["stem"], chunk_frames=chunk, dmax=dmax, stride=STRIDE,
        batch_size=batch_size, right_context=right,
    )


def test_every_window_has_the_same_width_and_context(tmp_path, cache):
    # Streams reset at different times, so a track's first chunk (padded on the left)
    # must be indistinguishable in shape from a mid-track one, or collate rejects it.
    ds = _dataset(tmp_path, cache, [900, 900, 600, 1200], chunk=40, dmax=30,
                  batch_size=2, right=5)
    batches = list(ds)
    assert len(batches) > 3
    want = (30 + 40 + 5) * STRIDE
    for batch in batches:
        assert batch.features.shape[-1] == want
        assert batch.context == 30
        assert batch.features.shape[:3] == (2, 1, N_BINS)


def test_a_tracks_first_chunk_is_left_padded_with_zeros(tmp_path, cache):
    # The pad exists so context is uniform; it must be zeros, since the scan relies on
    # the carry's NEG (not on the features) to make it contribute nothing.
    ds = _dataset(tmp_path, cache, [600, 600], chunk=40, dmax=30, batch_size=2)
    first = next(iter(ds))
    assert all(first.resets)
    assert torch.count_nonzero(first.features[:, :, :, : 30 * STRIDE]) == 0
    assert torch.count_nonzero(first.features[:, :, :, 30 * STRIDE :]) > 0


def test_reset_is_flagged_exactly_at_track_boundaries(tmp_path, cache):
    # One stream, so the batch rows are that stream's chunks in order: reset must fire
    # on the first chunk of each track and nowhere else.
    n, chunk = 600, 40
    ds = _dataset(tmp_path, cache, [n, n, n], chunk=chunk, dmax=30, batch_size=1)
    resets = [b.resets[0] for b in ds]
    per_track = (n // STRIDE) // chunk
    assert resets[0] is True
    assert [k for k, r in enumerate(resets) if r] == [
        k * per_track for k in range(len(resets) // per_track)
    ]


def test_gold_segments_tile_the_track_across_chunks(tmp_path, cache):
    # Each gold segment is assigned to exactly one chunk (the one holding its end), so
    # the per-chunk pieces must partition the whole-track gold, with no segment dropped
    # or double-counted -- the assumption the telescoping loss rests on.
    n, chunk, dmax = 600, 40, 30
    ds = _dataset(tmp_path, cache, [n], chunk=chunk, dmax=dmax, batch_size=1)
    per_track = (n // STRIDE) // chunk

    spans = []
    for c, batch in enumerate(ds):
        # window-local -> global: the window starts dmax before the chunk's own frames
        offset = c * chunk - dmax
        spans += [(i + offset, j + offset) for _, i, j in batch.golds[0]]

    reference = _Item("t0", n, tmp_path).reference_notes()
    from stemify.decoders.semicrf.gold import gold_segments

    whole = gold_segments(np.arange(n)[::STRIDE] * HOP_S, reference, dmax)
    covered = [(i, j) for _, i, j in whole if j <= per_track * chunk]

    assert spans == sorted(spans), "segments must arrive in track order"
    assert len(set(spans)) == len(spans), "a segment landed in two chunks"
    assert spans == covered, "per-chunk gold is not the whole-track gold, split"


def test_streams_interleave_distinct_tracks(tmp_path, cache):
    # Row b must be a coherent stream; two rows must never be the same track at once,
    # or the batch is scanning one track twice and the carry is meaningless.
    ds = _dataset(tmp_path, cache, [600] * 4, chunk=40, dmax=30, batch_size=2)
    for batch in ds:
        assert batch.streams == [0, 1]


def test_collate_rejects_mixed_widths(tmp_path, cache):
    ds = _dataset(tmp_path, cache, [600, 600], chunk=40, dmax=30, batch_size=1)
    a, b = list(ds)[:2]
    from stemify.training.datasets.chunked import Chunk

    wide = Chunk(
        np.zeros((1, N_BINS, 7)), np.arange(7) * HOP_S, a.contours[0], [], 30, 0,
        0, False
    )
    narrow = Chunk(
        a.features[0].numpy(), a.frame_times[0], a.contours[0], [], 30, 0, 1, False
    )
    with pytest.raises(ValueError, match="share a shape"):
        collate([wide, narrow])


def test_collate_rejects_mixed_margins(tmp_path, cache):
    # The scan runs one shared (context, right) range over every row, so a row with a
    # different margin would have its own frames read as context, or its context
    # scanned as chunk -- silently, since the widths still match.
    ds = _dataset(tmp_path, cache, [600, 600], chunk=40, dmax=30, batch_size=1)
    a = next(iter(ds))
    from stemify.training.datasets.chunked import Chunk

    lhs = Chunk(a.features[0].numpy(), a.frame_times[0], a.contours[0], [], 30, 0,
                0, False)
    rhs = Chunk(a.features[0].numpy(), a.frame_times[0], a.contours[0], [], 30, 5,
                1, False)
    with pytest.raises(ValueError, match=r"share \(context, right\)"):
        collate([lhs, rhs])


def _cues(frame_times, contour):
    """Deterministic pseudo-cues standing in for a model's head logits.

    Keyed to the *absolute* frame index, not the position within the window: a real
    encoder is translation-equivariant, so the same frame must get the same cue in
    whichever chunk it appears, or the chunked and whole-track scans are not being
    given the same problem."""
    k = np.round(frame_times / HOP_S)
    g = (np.abs(np.sin(k * 0.7)) * 0.98 + 0.01).astype(np.float32)
    v = (np.abs(np.cos(k * 0.3)) * 0.98 + 0.01).astype(np.float32)
    return Cues(contour=contour, onset=g, frame_times=frame_times, voicing=v)


def test_chunks_through_the_carry_equal_the_whole_track_log_partition(tmp_path, cache):
    # The end-to-end property: geometry from the dataset, scanned with the carry, must
    # reproduce the whole-track log Z. This is what ties the chunking to the objective.
    n_cqt, chunk, dmax = 900, 40, 30
    ds = _dataset(tmp_path, cache, [n_cqt], chunk=chunk, dmax=dmax, batch_size=1)
    dec = SemiCRFDecoder(dmax_s=dmax * HOP_S * STRIDE)
    dur_bucket = torch.arange(dmax + 1).clamp(max=dec.pot.dur.numel() - 1)

    batches = list(ds)
    n_dp = (n_cqt // STRIDE // chunk) * chunk

    # whole track, over exactly the frames the chunks cover
    contour = _NoOpCache(tmp_path).contour(None, "x")
    ft = np.arange(n_dp * STRIDE) * HOP_S
    whole_e = build_emissions(_cues(ft, contour), STRIDE)
    whole = batched_log_partition(dec.pot, pad_emissions([whole_e]), dmax, dur_bucket)

    carry, A = None, None
    for batch in batches:
        e = build_emissions(_cues(batch.frame_times[0], batch.contours[0]), STRIDE)
        A, carry = chunk_log_partition(
            dec.pot, pad_emissions([e]), dmax, dur_bucket, carry, batch.context
        )
    assert torch.allclose(A, whole, atol=1e-3), f"{A.item()} != {whole.item()}"
