"""Serve fixed-length training windows over cached ``TrackExample``s.

The batch-serving layer: preparation and caching of the per-track examples lives
in ``examples.py``; this only slices them into the fixed-length windows the model
trains on (the temporal stack needs a few seconds of context and a batch needs a
fixed length). Windows are read straight off the memmapped feature sidecar, so a
full-scale set never sits in RAM.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stemify.serde import TrackExample


class FrameWindowDataset(Dataset):
    """Fixed-length windows over cached ``TrackExample``s.

    Windows are enumerated up front (deterministic), so training order is a
    shuffle of a stable set rather than a moving target. A track shorter than one
    window yields a single zero-padded window rather than being dropped.

    A track is a *tuple* of cache paths, one per model input channel -- usually
    one, two for a mix-conditioned model. Each channel is its own cached CQT
    (content-addressed on its own audio file), so nothing here recomputes anything
    to add one; the windows are cut at the same offsets and stacked. **Targets and
    the window index come from channel 0 only**, which is the channel the labels
    describe -- see ``training/conditions.py``.
    """

    def __init__(
        self,
        tracks: Sequence[Sequence[Path]],
        heads: Iterable[str],
        window_frames: int = 256,
        hop_frames: int = 128,
    ) -> None:
        self.tracks = [tuple(t) for t in tracks]
        self.heads = list(heads)
        self.window = window_frames
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
        self._cache: dict[int, tuple[TrackExample, ...]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _examples(self, ei: int) -> tuple[TrackExample, ...]:
        # mmap: each cached TrackExample holds a memmap (no resident feature data)
        # plus the small targets/frame_times, so this cache stays cheap even
        # unbounded -- the big features are paged in per-window by the OS, shared
        # across workers. That sharing is why a second channel costs page cache
        # rather than a second resident copy per worker.
        if ei not in self._cache:
            self._cache[ei] = tuple(
                TrackExample.load(p, mmap=True) for p in self.tracks[ei]
            )
        return self._cache[ei]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ei, start = self._index[i]
        examples = self._examples(ei)
        n_bins = examples[0].features.shape[0]
        x = np.zeros((len(examples), n_bins, self.window), dtype=np.float32)
        for c, ex in enumerate(examples):
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


def collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack windows into (B, C, n_bins, W) and each target into (B, W)."""
    xs = torch.stack([x for x, _ in batch])
    heads = batch[0][1].keys()
    ys = {h: torch.stack([y[h] for _, y in batch]) for h in heads}
    return xs, ys
