"""Fixed-size chunks of tracks, in order, for semi-CRF training.

The obvious alternative -- yield whole tracks and pad a batch to the longest -- was
tried and removed: 16,617 frames padded to 23,355 on a real batch, 54 GiB of
activations at B=16, which does not fit. This yields fixed-size *chunks* instead, and
the alpha window carried between them (``forward.chunk_log_partition``) is what keeps
the objective the true whole-track ``log Z`` rather than a per-window approximation.
Untruncated whole-track backprop is still reachable -- it is ``SemiCRFLoss``'s
``detach_carry=False``, not a second dataset.

Chunks must arrive **in order, per track**, which a shuffling ``DataLoader`` cannot do.
The classic answer from RNN language modelling: run ``batch_size`` parallel *streams*,
each walking its own share of the tracks, and take one chunk from each per batch. So
this is an ``IterableDataset`` that emits whole ``ChunkBatch`` objects (use
``batch_size=None`` on the loader), not a map-style dataset of examples.

Each row carries a ``stream`` id and a ``reset`` flag rather than relying on position,
because with ``num_workers > 1`` consecutive batches come from *different* workers, so
row ``b`` of one batch is not the continuation of row ``b`` of the last. The trainer
keys its carries by ``stream``; see ``loss.py`` beside this.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stemify.decoders.semicrf.gold import gold_segments
from stemify.training.semicrf.tracks import (
    TrackInputs, channel_window, target_windows,
)

# Segmentation of one chunk in *window-local* DP indices: (label, start, end).
Segmentation = list[tuple[int, int, int]]


@dataclass
class Chunk:
    """One chunk of one track, with its context prepended.

    ``features`` spans ``[context_start, chunk_end + right_context)``; the DP scans
    only the chunk's own frames (``forward.chunk_log_partition(..., context=context)``)
    and reads the rest purely as context.

    No contour: after the pyin span features were removed, no potential reads one, and
    every recipe builds a voicing head -- so the scan never reaches the fallback that
    was a contour's last use here. A pitch track is an inference-time concern now.
    """

    features: np.ndarray       # (C, n_bins, window CQT frames)
    frame_times: np.ndarray    # (window CQT frames,) absolute seconds
    gold: Segmentation         # window-local DP indices, ends inside this chunk
    y: dict[str, np.ndarray]   # {head: (window CQT frames,)}, NaN where off the track
    stream: int                # which stream row this continues
    reset: bool                # first chunk of a track: the carry starts fresh


@dataclass
class ChunkBatch:
    """One chunk from each of ``batch_size`` streams, stacked.

    ``y`` is the only field that grows: a new head is a key, and ``collate`` stacks it
    without knowing the name. Everything else is fixed bookkeeping the scan needs, kept
    as named fields precisely so that adding to it takes a decision.
    """

    features: torch.Tensor          # (B, C, n_bins, window)
    y: dict[str, torch.Tensor]      # {head: (B, window)}, on the feature grid
    golds: list[Segmentation]
    frame_times: list[np.ndarray]
    context: int                    # DP frames of left context; set by the dataset
    right: int                      # DP frames the scan must not read; ditto
    streams: list[int]
    resets: list[bool]

    def __len__(self) -> int:
        return len(self.streams)


def collate(chunks: list[Chunk], context: int, right: int) -> ChunkBatch:
    """Stack chunks into a batch. No padding: every chunk is the same width, which is
    also what lets the scan be captured as a CUDA graph.

    ``context`` and ``right`` are passed in rather than read off the chunks. They are
    properties of how the *dataset* cut the track, identical for every chunk it emits,
    and copying a constant onto each row only to check the copies still agree is a
    round trip that invents the invariant it then guards.

    ``y`` stacks by key without this knowing any of them, so a future head (string,
    fret, velocity) costs nothing here."""
    widths = {c.features.shape for c in chunks}
    if len(widths) != 1:
        raise ValueError(f"chunks must share a shape, got {sorted(widths)}")
    heads = {tuple(sorted(c.y)) for c in chunks}
    if len(heads) != 1:
        raise ValueError(f"chunks must carry the same targets, got {sorted(heads)}")
    features = torch.from_numpy(
        np.ascontiguousarray(np.stack([c.features for c in chunks]))
    )
    return ChunkBatch(
        features=features,
        y={
            head: torch.from_numpy(np.stack([c.y[head] for c in chunks]))
            for head in heads.pop()
        },
        golds=[c.gold for c in chunks],
        frame_times=[c.frame_times for c in chunks],
        context=context,
        right=right,
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
        self, track: TrackInputs, chunk: int, dmax: int, stride: int, right: int,
        heads: Sequence[str] = (),
    ) -> None:
        self._examples = track.open(mmap=True)
        self._heads = heads
        self._chunk, self._dmax, self._stride = chunk, dmax, stride
        ft = self._examples[0].frame_times
        self._n_cqt = ft.size
        self._n_dp = self._n_cqt // stride
        # librosa's grid is exactly n * hop_length / sr, so the window's times -- including
        # the padded ones outside the track -- extrapolate exactly from one hop.
        self._hop = float(ft[1] - ft[0]) if ft.size > 1 else 0.01
        self._w_dp = dmax + chunk + right
        # Gold is built once over the whole track, then split by chunk, so the pieces
        # tile it exactly -- which is what makes the per-chunk losses telescope.
        self._gold = gold_segments(ft[::stride], track.notes, dmax)
        self.n_chunks = self._n_dp // chunk  # a ragged tail would break static shapes

    def __getitem__(
        self, c: int
    ) -> tuple[np.ndarray, np.ndarray, Segmentation, dict[str, np.ndarray]]:
        lo, hi = c * self._chunk, (c + 1) * self._chunk
        a_dp = lo - self._dmax                      # negative on a track's first chunk
        a, w = a_dp * self._stride, self._w_dp * self._stride

        # (C, n_bins, w) -- every input channel, zero-padded where the window runs off
        # either end of the track.
        window = channel_window(self._examples, a, w)
        times = (np.arange(a, a + w) * self._hop).astype(np.float64)

        # A gold segment is assigned to the chunk holding its END. Its start is then at
        # worst dmax earlier, so a dmax left context always covers it -- the same
        # constant the scan needs for its own candidate starts.
        gold = [(lab, i - a_dp, j - a_dp) for lab, i, j in self._gold if lo < j <= hi]
        # Targets span the whole window, not just the chunk's own frames, so they stay
        # index-aligned with the features and the model's output; the loss slices the
        # scanned range out. Channel 0 only -- the channel the labels describe.
        y = target_windows(self._examples[0], self._heads, a, w)
        return window, times, gold, y


class ChunkDataset(IterableDataset[ChunkBatch]):
    """Chunks of tracks over ``batch_size`` parallel streams, in track order.

    Streams are independent: each takes every ``n``-th track, walks it chunk by chunk,
    then moves to its next track and flags ``reset``. A stream that runs out stops the
    whole worker, so a little data is dropped each epoch -- the alternative is a ragged
    final batch, and static shapes are worth more than the tail. Round-robin assignment
    keeps stream durations close enough that the loss is small.

    With ``num_workers > 1`` each worker owns a disjoint slice of the tracks and emits
    complete batches of its own; ``stream`` ids stay globally unique so the trainer's
    carries never collide.

    **Domain randomisation.** ``items`` may come from several input conditions at once,
    and nothing below this line knows which condition a track came from -- a
    ``TrackInputs`` is feature paths plus notes, and ``resolve()`` guarantees every
    condition in a run has the same channel width, so the windows stack identically.
    Building the item list is therefore the recipe's job, not this class's; it used to
    take a ``(dataset, condition, cache)`` triple and resolve one condition itself,
    which is what limited the structured objective to a single condition while the frame
    path trained on the union."""

    def __init__(
        self,
        items: Sequence[TrackInputs],
        *,
        chunk_frames: int,
        dmax: int,
        stride: int,
        batch_size: int,
        right_context: int = 0,
        heads: Sequence[str] = (),
        seed: int = 0,
    ) -> None:
        """``heads`` are the per-frame targets served beside each chunk, for the
        auxiliary losses stacked on the NLL. Empty by default: the structured
        objective reads notes, so a head appears here only once something scores it
        per frame.

        ``items`` are **shuffled once** here, and that shuffle is load-bearing rather
        than cosmetic. Streams take ``items[b::batch_size]`` and advance in near
        lockstep, so the incoming order decides what a batch contains:

        - concatenated by condition (``[...clean..., ...separated...]``) every stream
          sits in the clean block at the same time, so batches come out
          condition-*homogeneous* -- and the TCN's BatchNorm then sees one condition's
          statistics at a time, which is the thing domain randomisation exists to avoid;
        - interleaved (``clean, separated, clean, ...``) stream ``b`` only ever draws
          items of one parity, so each stream is permanently one condition.

        Both train on the union and report the right ``--sources`` while never mixing
        conditions in a batch. Seeded so a run is still reproducible.
        """
        self.heads = list(heads)
        self._items = list(items)
        random.Random(seed).shuffle(self._items)
        self.chunk_frames = chunk_frames
        self.dmax = dmax
        self.stride = stride
        self.batch_size = batch_size
        # Frames past the chunk's end, scanned by nobody: they exist so the encoder's
        # receptive field does not see zero-padding at the chunk's right edge. The left
        # edge needs no such margin -- dmax of real context already covers it.
        self.right_context = right_context

    def _shard(self) -> tuple[list[TrackInputs], int]:
        """This worker's tracks, and the id its first stream takes."""
        info = get_worker_info()
        if info is None:
            return self._items, 0
        return self._items[info.id :: info.num_workers], info.id * self.batch_size

    def __iter__(self) -> Iterator[ChunkBatch]:
        items, base = self._shard()
        streams = [items[b :: self.batch_size] for b in range(self.batch_size)]
        if not all(streams):
            # Used to `return` here, which emitted nothing and said nothing: the epoch
            # then averaged over zero batches and the trainer died several frames later
            # on a missing "total" key. The condition is a plain misconfiguration --
            # a val set smaller than the batch is the usual way in -- so it is worth
            # naming both numbers and the fix.
            raise ValueError(
                f"{len(items)} track(s) cannot fill {self.batch_size} parallel streams: "
                "every stream needs at least one track, and a stream that starts empty "
                "would make the batch width inconsistent. Lower batch_size to at most "
                f"{len(items)}, or give this split more tracks."
            )
        cursors = [0] * self.batch_size          # index into this stream's track list
        opened: list[_Track | None] = [None] * self.batch_size
        chunk_no = [0] * self.batch_size

        while True:
            row: list[Chunk] = []
            for b in range(self.batch_size):
                reset = False
                # One name for the stream's current track instead of re-reading
                # `opened[b]` four times: the loop's whole job is to leave it open, so
                # the value that falls out of it is the one the chunk is cut from.
                track = opened[b]
                while track is None or chunk_no[b] >= track.n_chunks:
                    if track is not None:
                        cursors[b] += 1
                    if cursors[b] >= len(streams[b]):
                        return  # this stream is done; stop the whole worker
                    track = opened[b] = _Track(
                        streams[b][cursors[b]], self.chunk_frames, self.dmax,
                        self.stride, self.right_context, self.heads,
                    )
                    chunk_no[b], reset = 0, True
                feats, ft, gold, y = track[chunk_no[b]]
                chunk_no[b] += 1
                row.append(
                    Chunk(feats, ft, gold=gold, y=y, stream=base + b, reset=reset)
                )
            yield collate(row, context=self.dmax, right=self.right_context)
