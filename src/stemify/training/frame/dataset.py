"""Serve fixed-length training windows over cached ``CachedItem``s.

The batch-serving layer: preparation and caching of the per-track examples lives
in ``examples.py``; this only slices them into the fixed-length windows the model
trains on (the temporal stack needs a few seconds of context and a batch needs a
fixed length). Windows are read straight off the memmapped feature sidecar, so a
full-scale set never sits in RAM -- but see ``_examples`` for the part of a
``CachedItem`` that *is* resident, and why the open-track cache is bounded.
"""

from collections.abc import Iterable, Sequence
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stemify.core.features import SILENCE_DB
from stemify.core.serde import CachedItem

# How many tracks' resident state one dataset keeps open. Sized to be negligible per
# worker rather than to hit some ratio -- see ``_examples`` for why a larger cache
# would buy nothing.
CACHE_TRACKS = 64


class FrameWindowDataset(Dataset[Any]):
    """Fixed-length windows over cached ``CachedItem``s.

    Windows are enumerated up front (deterministic), so training order is a
    shuffle of a stable set rather than a moving target. A track shorter than one
    window yields a single zero-padded window rather than being dropped.

    A track is a *tuple* of cache paths, one per model input channel -- usually
    one, two for a mix-conditioned model. Each channel is its own cached CQT
    (content-addressed on its own audio file), so nothing here recomputes anything
    to add one; the windows are cut at the same offsets and stacked. **Targets and
    the window index come from channel 0 only**, which is the channel the labels
    describe -- see ``core/conditions.py``.
    """

    def __init__(
        self,
        tracks: Sequence[Sequence[Path]],
        heads: Iterable[str],
        window_frames: int = 256,
        hop_frames: int = 128,
        stem_dropout: float = 0.0,
    ) -> None:
        """``stem_dropout`` is the per-window probability of silencing each *context*
        channel, so a bichannel model also learns to transcribe from the bass alone.

        Channel 0 is never dropped: it is the audio the labels describe, and a window
        with no bass is a question with no answer rather than a harder example.

        Zero by default, and it must stay that way for the existing arms -- it changes
        the input distribution, so a run that gains it is not comparable to one that
        did not have it.
        """
        if not 0.0 <= stem_dropout <= 1.0:
            raise ValueError(f"stem_dropout must be in [0, 1], got {stem_dropout}")
        self.tracks = [tuple(t) for t in tracks]
        self.heads = list(heads)
        self.window = window_frames
        self.stem_dropout = stem_dropout
        self._index: list[tuple[int, int]] = []
        for ei, paths in enumerate(self.tracks):
            counts = [
                int(np.load(p, allow_pickle=False)["frame_times"].shape[0])
                for p in paths
            ]
            # A track's channels are renderings of the same music on the same CQT
            # hop, so their frame counts match exactly (verified across Slakh's
            # stem/mix/demucs conditions). A mismatch therefore means the channels
            # were paired wrongly, not that one file is slightly longer -- and that
            # failure is invisible downstream: the labels still fit channel 0, the
            # loss still falls, and the model just never learns from a mixture that
            # belongs to a different song.
            if len(set(counts)) > 1:
                raise ValueError(
                    f"channels of {paths[0].name} disagree on length ({counts}); "
                    "they should be the same track through different transforms"
                )
            n_frames = counts[0]
            if n_frames <= window_frames:
                self._index.append((ei, 0))
            else:
                for start in range(0, n_frames - window_frames + 1, hop_frames):
                    self._index.append((ei, start))
        self._cache: dict[int, tuple[CachedItem, ...]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _examples(self, ei: int) -> tuple[CachedItem, ...]:
        # Features are memmapped, so they cost page cache the OS shares across
        # workers. Targets and frame_times are *not*: they are read resident out of
        # the npz, and every worker is a fork with its own copy.
        #
        # Hence the bound. An entry is well under a megabyte, but the train set is one
        # per track per condition, so an unbounded cache converges on the whole corpus
        # in each worker -- which is enough to OOM a host well before the GPU notices.
        # Nothing is lost by evicting: `shuffle=True` leaves no locality to exploit, so
        # the cache only ever spares a repeat open of a small file.
        cached = self._cache.get(ei)
        if cached is None:
            cached = tuple(CachedItem.load(p, mmap=True) for p in self.tracks[ei])
            if len(self._cache) >= CACHE_TRACKS:
                self._cache.pop(next(iter(self._cache)))  # FIFO; dicts keep order
            self._cache[ei] = cached
        return cached

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ei, start = self._index[i]
        examples = self._examples(ei)
        n_bins = examples[0].features.shape[0]
        # SILENCE_DB, not zero: features are dB relative to the track's peak, so a
        # zero-filled tail is full-scale energy rather than an absent one. Only reached
        # for tracks shorter than one window, but the fill has to mean silence there
        # too -- and stem dropout below uses the same constant for the same reason.
        x = np.full((len(examples), n_bins, self.window), SILENCE_DB, dtype=np.float32)
        for c, ex in enumerate(examples):
            # Channel 0 is the bass and is never dropped; see __init__. `torch.rand`
            # rather than numpy's global RNG because DataLoader workers are forks:
            # PyTorch reseeds each worker's torch generator (base_seed + worker_id)
            # while numpy's is inherited identically, so a numpy draw here would make
            # every worker silence the same windows.
            if c and self.stem_dropout and float(torch.rand(())) < self.stem_dropout:
                continue                      # already SILENCE_DB from the fill
            # np.asarray copies just this window out of the memmap into a real array.
            feat = np.asarray(ex.features[:, start:start + self.window])
            x[c, :, : feat.shape[1]] = feat
        y: dict[str, torch.Tensor] = {}
        for h in self.heads:
            t = examples[0].targets[h][start:start + self.window]
            padded = np.zeros(self.window, dtype=np.float32)
            padded[: t.shape[0]] = t
            y[h] = torch.from_numpy(padded)
        return torch.from_numpy(x), y  # (C, n_bins, W), {head: (W,)}


@dataclass
class FrameBatch:
    """A minibatch of windows and their per-frame targets.

    A named type rather than the bare ``(x, y)`` tuple this used to be, because the
    training loop asks a batch for its size and ``len((x, y))`` answers **2**. That
    wrong answer is why ``batch_size`` used to live on the ``Loss`` -- a property of
    the batch, implemented on its consumer, once per objective.
    """

    x: torch.Tensor              # (B, C, n_bins, W)
    y: dict[str, torch.Tensor]   # {head: (B, W)}

    def __len__(self) -> int:
        return int(self.x.shape[0])


def collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> FrameBatch:
    """Stack windows into (B, C, n_bins, W) and each target into (B, W)."""
    xs = torch.stack([x for x, _ in batch])
    heads = batch[0][1].keys()
    ys = {h: torch.stack([y[h] for _, y in batch]) for h in heads}
    return FrameBatch(xs, ys)
