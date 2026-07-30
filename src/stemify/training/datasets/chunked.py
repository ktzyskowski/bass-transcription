"""Fixed-size chunks of tracks, in order, for truncated-BPTT semi-CRF training.

``SequenceDataset`` yields whole tracks and pads a batch to the longest -- 16,617
frames padded to 23,355 on a real batch, and 54 GiB of activations at B=16, which does
not fit. This yields fixed-size *chunks* instead, and the alpha window carried between
them (``forward.chunk_log_partition``) is what keeps the objective the true whole-track
``log Z`` rather than a per-window approximation.

Chunks must arrive **in order, per track**, which a shuffling ``DataLoader`` cannot do.
The classic answer from RNN language modelling: run ``batch_size`` parallel *streams*,
each walking its own share of the tracks, and take one chunk from each per batch. So
this is an ``IterableDataset`` that emits whole ``ChunkBatch`` objects (use
``batch_size=None`` on the loader), not a map-style dataset of examples.

Each row carries a ``stream`` id and a ``reset`` flag rather than relying on position,
because with ``num_workers > 1`` consecutive batches come from *different* workers, so
row ``b`` of one batch is not the continuation of row ``b`` of the last. The trainer
keys its carries by ``stream``; see ``losses/segmental.py``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stemify.cache import Cache
from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.decoders.semicrf.gold import gold_segments
from stemify.notes import NoteEvent
from stemify.pitch import PyinPitch
from stemify.serde import TrackExample
from stemify.transforms import AudioTransform, Identity

# Segmentation of one chunk in *window-local* DP indices: (label, start, end).
Segmentation = list[tuple[int, int, int]]


@dataclass
class Chunk:
    """One chunk of one track, with its context prepended.

    ``features`` spans ``[context_start, chunk_end + right_context)``; the DP scans
    only the chunk's own frames (``forward.chunk_log_partition(..., context=context)``)
    and reads the rest purely as context."""

    features: np.ndarray       # (n_bins, window CQT frames)
    frame_times: np.ndarray    # (window CQT frames,) absolute seconds
    contour: Contour           # the whole track's pyin contour (searchsorted per use)
    gold: Segmentation         # window-local DP indices, ends inside this chunk
    context: int               # DP frames of left context before the chunk
    stream: int                # which stream row this continues
    reset: bool                # first chunk of a track: the carry starts fresh


@dataclass
class ChunkBatch:
    """One chunk from each of ``batch_size`` streams, stacked."""

    features: torch.Tensor        # (B, 1, n_bins, window)
    frame_times: list[np.ndarray]
    contours: list[Contour]
    golds: list[Segmentation]
    context: int                  # uniform across the batch
    streams: list[int]
    resets: list[bool]

    def __len__(self) -> int:
        return len(self.streams)


def collate(chunks: list[Chunk]) -> ChunkBatch:
    """Stack chunks into a batch. No padding: every chunk is the same width, which is
    also what lets the scan be captured as a CUDA graph."""
    widths = {c.features.shape[1] for c in chunks}
    if len(widths) != 1:
        raise ValueError(f"chunks must share a width, got {sorted(widths)}")
    contexts = {c.context for c in chunks}
    if len(contexts) != 1:
        raise ValueError(
            f"chunks in a batch must share a context, got {sorted(contexts)}; "
            "streams have desynchronised"
        )
    features = torch.from_numpy(
        np.ascontiguousarray(np.stack([c.features for c in chunks]))
    ).unsqueeze(1)
    return ChunkBatch(
        features=features,
        frame_times=[c.frame_times for c in chunks],
        contours=[c.contour for c in chunks],
        golds=[c.gold for c in chunks],
        context=contexts.pop(),
        streams=[c.stream for c in chunks],
        resets=[c.reset for c in chunks],
    )


class _Track:
    """Walks one track's chunks in order, holding its mmap open across them.

    All geometry is in DP frames (the ``STRIDE``-decimated grid the scan runs on) and
    converted to CQT frames only for the feature slice, so ``context`` lines up with
    ``dmax`` without a rounding step.

    Every window is exactly ``dmax + chunk + right`` DP frames, zero-padded where the
    track does not reach -- at its very start, and past its end for the right margin.
    Uniform width and a uniform ``context`` are what let chunks from streams that reset
    at different times share a batch, and what keep the scan's shapes static.

    Padding at the track's start is safe rather than merely harmless: a fresh track
    enters with ``initial_window``, which is ``NEG`` everywhere except ``START`` at
    exactly the position the real frame 0 lands on, so the fake frames contribute
    ``exp(NEG - max) == 0``."""

    def __init__(
        self, path: Path, contour: Contour, notes: list[NoteEvent],
        chunk: int, dmax: int, stride: int, right: int,
    ) -> None:
        self._ex = TrackExample.load(path, mmap=True)
        self.contour = contour
        self._chunk, self._dmax, self._stride = chunk, dmax, stride
        ft = self._ex.frame_times
        self._n_cqt = ft.size
        self._n_dp = self._n_cqt // stride
        # librosa's grid is exactly n * hop_length / sr, so the window's times -- including
        # the padded ones outside the track -- extrapolate exactly from one hop.
        self._hop = float(ft[1] - ft[0]) if ft.size > 1 else 0.01
        self._w_dp = dmax + chunk + right
        # Gold is built once over the whole track, then split by chunk, so the pieces
        # tile it exactly -- which is what makes the per-chunk losses telescope.
        self._gold = gold_segments(ft[::stride], notes, dmax)
        self.n_chunks = self._n_dp // chunk  # a ragged tail would break static shapes

    def __getitem__(self, c: int) -> tuple[np.ndarray, np.ndarray, Segmentation]:
        lo, hi = c * self._chunk, (c + 1) * self._chunk
        a_dp = lo - self._dmax                      # negative on a track's first chunk
        a, w = a_dp * self._stride, self._w_dp * self._stride

        window = np.zeros((self._ex.features.shape[0], w), dtype=np.float32)
        src_lo, src_hi = max(0, a), min(self._n_cqt, a + w)
        if src_lo < src_hi:
            window[:, src_lo - a : src_hi - a] = self._ex.features[:, src_lo:src_hi]
        times = (np.arange(a, a + w) * self._hop).astype(np.float64)

        # A gold segment is assigned to the chunk holding its END. Its start is then at
        # worst dmax earlier, so a dmax left context always covers it -- the same
        # constant the scan needs for its own candidate starts.
        gold = [(lab, i - a_dp, j - a_dp) for lab, i, j in self._gold if lo < j <= hi]
        return window, times, gold


class ChunkedSequenceDataset(IterableDataset):
    """Chunks of tracks over ``batch_size`` parallel streams, in track order.

    Streams are independent: each takes every ``n``-th track, walks it chunk by chunk,
    then moves to its next track and flags ``reset``. A stream that runs out stops the
    whole worker, so a little data is dropped each epoch -- the alternative is a ragged
    final batch, and static shapes are worth more than the tail. Round-robin assignment
    keeps stream durations close enough that the loss is small.

    With ``num_workers > 1`` each worker owns a disjoint slice of the tracks and emits
    complete batches of its own; ``stream`` ids stay globally unique so the trainer's
    carries never collide."""

    def __init__(
        self,
        dataset: SlakhLikeDataset,
        cache: Cache,
        *,
        chunk_frames: int,
        dmax: int,
        stride: int,
        batch_size: int,
        right_context: int = 0,
        transform: AudioTransform | None = None,
    ) -> None:
        transform = transform or Identity()
        estimator = PyinPitch()
        cache.ensure_features(dataset, transform)
        self._items: list[tuple[Path, Contour, list[NoteEvent]]] = []
        for item in dataset:
            audio = transform.apply(item.audio())
            self._items.append(
                (cache.feature_path(audio), cache.contour(estimator, audio),
                 item.reference_notes())
            )
        self.chunk_frames = chunk_frames
        self.dmax = dmax
        self.stride = stride
        self.batch_size = batch_size
        # Frames past the chunk's end, scanned by nobody: they exist so the encoder's
        # receptive field does not see zero-padding at the chunk's right edge. The left
        # edge needs no such margin -- dmax of real context already covers it.
        self.right_context = right_context

    def _shard(self) -> tuple[list, int]:
        """This worker's tracks, and the id its first stream takes."""
        info = get_worker_info()
        if info is None:
            return self._items, 0
        return self._items[info.id :: info.num_workers], info.id * self.batch_size

    def __iter__(self) -> Iterator[ChunkBatch]:
        items, base = self._shard()
        streams = [items[b :: self.batch_size] for b in range(self.batch_size)]
        if not all(streams):
            return  # fewer tracks than streams: nothing coherent to emit
        cursors = [0] * self.batch_size          # index into this stream's track list
        opened: list[_Track | None] = [None] * self.batch_size
        chunk_no = [0] * self.batch_size

        while True:
            row: list[Chunk] = []
            for b in range(self.batch_size):
                reset = False
                while opened[b] is None or chunk_no[b] >= opened[b].n_chunks:
                    if opened[b] is not None:
                        cursors[b] += 1
                    if cursors[b] >= len(streams[b]):
                        return  # this stream is done; stop the whole worker
                    path, contour, notes = streams[b][cursors[b]]
                    opened[b] = _Track(
                        path, contour, notes, self.chunk_frames, self.dmax,
                        self.stride, self.right_context,
                    )
                    chunk_no[b], reset = 0, True
                feats, ft, gold = opened[b][chunk_no[b]]
                chunk_no[b] += 1
                row.append(
                    Chunk(feats, ft, contour=opened[b].contour, gold=gold,
                          context=self.dmax, stream=base + b, reset=reset)
                )
            yield collate(row)
