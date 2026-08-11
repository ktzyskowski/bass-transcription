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
a given *row* of one batch is not the continuation of that row of the last. The trainer
keys its carries by ``stream``; see ``loss.py`` beside this.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import count

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stemify.audio.features import SILENCE_DB
from stemify.semicrf.fit.geometry import ChunkGeometry
from stemify.semicrf.segmentation import LocalSegmentation, TrackSegmentation
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.semicrf.fit.tracks import (
    TrainItem, channel_window, target_windows,
)




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
    gold: LocalSegmentation    # this window's own DP indices; ends inside the chunk
    y: dict[str, np.ndarray]   # {head: (window CQT frames,)}, NaN where off the track
    stream: int                # which stream row this continues
    reset: bool                # the carry starts fresh here: a track's or a segment's
    prev: int                  # label of the gold segment preceding this chunk's first


@dataclass
class ChunkBatch:
    """One chunk from each of ``batch_size`` streams, stacked.

    ``y`` is the only field that grows: a new head is a key, and ``collate`` stacks it
    without knowing the name. Everything else is fixed bookkeeping the scan needs, kept
    as named fields precisely so that adding to it takes a decision.
    """

    features: torch.Tensor          # (B, C, n_bins, window)
    y: dict[str, torch.Tensor]      # {head: (B, window)}, on the feature grid
    golds: list[LocalSegmentation]
    frame_times: list[np.ndarray]
    geometry: ChunkGeometry         # how the dataset cut the track; see `fit.geometry`
    streams: list[int]
    resets: list[bool]
    prevs: list[int]                # per row: what precedes its first gold segment

    def __len__(self) -> int:
        return len(self.streams)


def collate(chunks: list[Chunk], geometry: ChunkGeometry) -> ChunkBatch:
    """Stack chunks into a batch. No padding: every chunk is the same width, which is
    also what lets the scan be captured as a CUDA graph.

    ``geometry`` is passed in rather than read off the chunks. It describes how the
    *dataset* cut the track, identically for every chunk it emits, and copying a
    constant onto each row only to check the copies still agree is a round trip that
    invents the invariant it then guards.

    ``y`` stacks by key without this knowing any of them, so a future head (string,
    fret, velocity) costs nothing here."""
    widths = {chunk.features.shape for chunk in chunks}
    if len(widths) != 1:
        raise ValueError(f"chunks must share a shape, got {sorted(widths)}")
    heads = {tuple(sorted(chunk.y)) for chunk in chunks}
    if len(heads) != 1:
        raise ValueError(f"chunks must carry the same targets, got {sorted(heads)}")
    features = torch.from_numpy(
        np.ascontiguousarray(np.stack([chunk.features for chunk in chunks]))
    )
    return ChunkBatch(
        features=features,
        y={
            head: torch.from_numpy(np.stack([chunk.y[head] for chunk in chunks]))
            for head in heads.pop()
        },
        golds=[chunk.gold for chunk in chunks],
        frame_times=[chunk.frame_times for chunk in chunks],
        geometry=geometry,
        streams=[chunk.stream for chunk in chunks],
        resets=[chunk.reset for chunk in chunks],
        prevs=[chunk.prev for chunk in chunks],
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
        self, track: TrainItem, chunk: int, dmax: int, stride: int, right: int,
        heads: Sequence[str] = (), drop_context: bool = False,
        labels: LabelSet = BINARY,
    ) -> None:
        self._examples = track.open(mmap=True)
        self._heads = heads
        # Whole-track rather than per-chunk, because that is the case it simulates: a
        # user uploads an isolated bass and *every* chunk of it lacks the mixture. Per
        # chunk would also make the carry cross a domain change mid-track, which is a
        # second thing to explain in any run that used it.
        self._drop_context = drop_context
        self._chunk, self._dmax, self._stride = chunk, dmax, stride
        # Left context is `dmax`: the scan's candidate starts reach exactly that far
        # back, so a window carrying less would cut a representable span in half.
        self.geometry = ChunkGeometry(
            context=dmax, chunk=chunk, right=right, stride=stride
        )
        ft = self._examples[0].frame_times
        self._n_cqt = ft.size
        self._n_dp = self._n_cqt // stride
        # librosa's grid is exactly n * hop_length / sr, so the window's times -- including
        # the padded ones outside the track -- extrapolate exactly from one hop.
        self._hop = float(ft[1] - ft[0]) if ft.size > 1 else 0.01
        # Gold is built once over the whole track, then split by chunk, so the pieces
        # tile it exactly -- which is what makes the per-chunk losses telescope.
        self._gold = gold_segments(ft[::stride], track.notes, dmax, labels)
        self._start = labels.START
        self.n_chunks = self._n_dp // chunk  # a ragged tail would break static shapes

    def __getitem__(
        self, c: int
    ) -> tuple[np.ndarray, np.ndarray, LocalSegmentation, dict[str, np.ndarray], int]:
        lo, hi = c * self._chunk, (c + 1) * self._chunk
        # Origin is negative on a track's first chunk: the window begins before the
        # track does and is zero-padded there.
        frame = self.geometry.at(lo - self.geometry.context)
        a, w = frame.cqt_window()

        # (C, n_bins, w) -- every input channel, silence-padded where the window runs
        # off either end of the track.
        window = channel_window(self._examples, a, w)
        if self._drop_context:
            window[1:] = SILENCE_DB      # channel 0 is the bass and always stays
        times = (np.arange(a, a + w) * self._hop).astype(np.float64)

        # A gold segment is assigned to the chunk holding its END. Its start is then at
        # worst dmax earlier, so a dmax left context always covers it -- the same
        # constant the scan needs for its own candidate starts.
        keep = [k for k, (_, _, end) in enumerate(self._gold) if lo < end <= hi]
        # Track DP frames become this window's own here -- the one coordinate change
        # in the pipeline, and the seam both recorded bugs in this area came through.
        gold = frame.localise_all(TrackSegmentation([self._gold[k] for k in keep]))
        # What the *previous* chunk ended on. Gold tiles the track contiguously, so the
        # segment before this chunk's first is simply the one before it in that tiling
        # -- and on a track's first chunk there is none, which is the only place START
        # genuinely belongs. Scoring every chunk's first segment against START instead
        # was a real defect: `log Z` applies the true transition, so the two halves of
        # the NLL disagreed by ~0.9 nats per chunk boundary once transitions trained.
        prev = self._gold[keep[0] - 1][0] if keep and keep[0] > 0 else self._start
        # Targets span the whole window, not just the chunk's own frames, so they stay
        # index-aligned with the features and the model's output; the loss slices the
        # scanned range out. Channel 0 only -- the channel the labels describe.
        y = target_windows(self._examples[0], self._heads, a, w)
        return window, times, gold, y, prev


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
    ``TrainItem`` is feature paths plus notes, and ``resolve()`` guarantees every
    condition in a run has the same channel width, so the windows stack identically.
    Building the item list is therefore the recipe's job, not this class's; it used to
    take a ``(dataset, condition, cache)`` triple and resolve one condition itself,
    which is what limited the structured objective to a single condition while the frame
    path trained on the union."""

    def __init__(
        self,
        items: Sequence[TrainItem],
        *,
        chunk_frames: int,
        dmax: int,
        stride: int,
        batch_size: int,
        right_context: int = 0,
        heads: Sequence[str] = (),
        seed: int = 0,
        stem_dropout: float = 0.0,
        labels: LabelSet = BINARY,
        segment_chunks: int = 0,
    ) -> None:
        """``heads`` are the per-frame targets served beside each chunk, for the
        auxiliary losses stacked on the NLL. Empty by default: the structured
        objective reads notes, so a head appears here only once something scores it
        per frame.

        ``items`` are **shuffled once** here, and that shuffle is load-bearing rather
        than cosmetic. Streams take ``items[row::batch_size]`` and advance in near
        lockstep, so the incoming order decides what a batch contains:

        - concatenated by condition (``[...clean..., ...separated...]``) every stream
          sits in the clean block at the same time, so batches come out
          condition-*homogeneous* -- and the TCN's BatchNorm then sees one condition's
          statistics at a time, which is the thing domain randomisation exists to avoid;
        - interleaved (``clean, separated, clean, ...``) stream ``row`` only ever draws
          items of one parity, so each stream is permanently one condition.

        Both train on the union and report the right ``--sources`` while never mixing
        conditions in a batch. Seeded so a run is still reproducible.

        ``segment_chunks`` breaks each track into independent runs of that many chunks,
        raising ``reset`` at each one's first chunk; ``0`` carries alpha across a whole
        track, which is what every run before this did.

        It exists to make the gradient exact. A carry passed *undetached* is the only
        way ``d(chunk k)/d(theta)`` reaches back through ``alpha_{k-1}``, and without it
        the loop's step is not a derivative of the loss it reports -- measured cosine
        0.9999 at initialisation falling to -0.02 after sixty steps, i.e. an ascent
        direction. But an undetached carry also means one backward per unbroken chain,
        so the chain length is the dial: shorter chains buy optimiser steps and bounded
        activation memory, longer ones buy an objective closer to the whole track's.

        Chunks stay fixed-width and every stream resets on the same chunk index, so the
        batch stays synchronised and one backward covers every row.
        """
        if not 0.0 <= stem_dropout <= 1.0:
            raise ValueError(f"stem_dropout must be in [0, 1], got {stem_dropout}")
        if segment_chunks < 0:
            raise ValueError(
                f"segment_chunks must be >= 0, got {segment_chunks}; 0 means carry "
                "alpha across the whole track"
            )
        self.segment_chunks = segment_chunks
        self.heads = list(heads)
        self._items = list(items)
        random.Random(seed).shuffle(self._items)
        self.seed = seed
        # Per-track probability of silencing every context channel, so a bichannel
        # model also learns to transcribe the bass alone. Zero by default: it changes
        # the input distribution, so an arm that gains it is not comparable to one
        # that did not.
        self.stem_dropout = stem_dropout
        # The alphabet gold is written in. It has to be the scorer's: a gold built
        # under BINARY labels every note 0, and under a pitch alphabet 0 is the
        # lowest representable semitone -- so the mismatch trains every note toward
        # E1 without changing a single shape.
        self.labels = labels
        self.chunk_frames = chunk_frames
        self.dmax = dmax
        self.stride = stride
        self.batch_size = batch_size
        # Frames past the chunk's end, scanned by nobody: they exist so the encoder's
        # receptive field does not see zero-padding at the chunk's right edge. The left
        # edge needs no such margin -- dmax of real context already covers it.
        self.right_context = right_context
        # One geometry for every chunk this dataset emits, and the same one each
        # `_Track` cuts with -- so the batch cannot describe a cutting that differs
        # from the one that produced it.
        self.geometry = ChunkGeometry(
            context=dmax, chunk=chunk_frames, right=right_context, stride=stride
        )

    def _shard(self) -> tuple[list[TrainItem], int]:
        """This worker's tracks, and the id its first stream takes."""
        info = get_worker_info()
        if info is None:
            return self._items, 0
        if self.segment_chunks:
            # Each worker counts its *own* batches while the loop interleaves them, so
            # the resets land at global positions the trainer's window never steps on --
            # and a row that does not reset where the backward falls carries a graph
            # across it and reads it freed. Same failure that keying on the stream's
            # chunk index produced, one level up, so it is refused rather than trusted
            # to stay unreachable: today only `num_workers=0` is wired, and the crash
            # would arrive twenty minutes into whichever run first changed that.
            raise NotImplementedError(
                f"segment_chunks={self.segment_chunks} needs num_workers=0. Chain "
                "resets are counted in batches so they align with the trainer's "
                f"accumulation window, and {info.num_workers} workers each count their "
                "own, so the two cannot agree."
            )
        return self._items[info.id :: info.num_workers], info.id * self.batch_size

    def __iter__(self) -> Iterator[ChunkBatch]:
        items, base = self._shard()
        streams = [items[row :: self.batch_size] for row in range(self.batch_size)]
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
        # Its own generator, seeded per worker: workers are forks that would otherwise
        # inherit one stream and drop the mixture on the *same* tracks in each. Derived
        # from `base` (worker id * batch_size) so it is still reproducible.
        rng = random.Random(self.seed + 1_000_003 * (base + 1))

        for batch_no in count():
            # **Counted in batches, not in each stream's chunk index.** The trainer's
            # accumulation window is a batch count, and the two have to agree exactly:
            # one backward covers the whole window, so a chain that straddles the
            # boundary has its second half reach into a graph the first half's backward
            # already freed. Streams sit at different positions in tracks of different
            # lengths, so `chunk_no[row] % segment_chunks` desynchronises the rows from
            # each other *and* from the trainer within one epoch -- which is a crash
            # partway through training, not a wrong number.
            boundary = bool(self.segment_chunks) and batch_no % self.segment_chunks == 0
            chunks: list[Chunk] = []
            for row in range(self.batch_size):
                reset = boundary
                # One name for the stream's current track instead of re-reading
                # `opened[row]` four times: the loop's whole job is to leave it open, so
                # the value that falls out of it is the one the chunk is cut from.
                track = opened[row]
                while track is None or chunk_no[row] >= track.n_chunks:
                    if track is not None:
                        cursors[row] += 1
                    if cursors[row] >= len(streams[row]):
                        return  # this stream is done; stop the whole worker
                    track = opened[row] = _Track(
                        streams[row][cursors[row]], self.chunk_frames, self.dmax,
                        self.stride, self.right_context, self.heads,
                        drop_context=rng.random() < self.stem_dropout,
                        labels=self.labels,
                    )
                    chunk_no[row], reset = 0, True
                feats, frame_times, gold, y, prev = track[chunk_no[row]]
                chunk_no[row] += 1
                chunks.append(
                    Chunk(feats, frame_times, gold=gold, y=y, stream=base + row,
                          reset=reset, prev=prev)
                )
            yield collate(chunks, self.geometry)
