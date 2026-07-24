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
    """

    def __init__(
        self,
        example_paths: Sequence[Path],
        heads: Iterable[str],
        window_frames: int = 256,
        hop_frames: int = 128,
    ) -> None:
        self.paths = list(example_paths)
        self.heads = list(heads)
        self.window = window_frames
        self._index: list[tuple[int, int]] = []
        for ei, p in enumerate(self.paths):
            n_frames = int(np.load(p, allow_pickle=False)["frame_times"].shape[0])
            if n_frames <= window_frames:
                self._index.append((ei, 0))
            else:
                for start in range(0, n_frames - window_frames + 1, hop_frames):
                    self._index.append((ei, start))
        self._cache: dict[int, TrackExample] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _example(self, ei: int) -> TrackExample:
        # mmap: the cached TrackExample holds a memmap (no resident feature data) plus
        # the small targets/frame_times, so this cache stays cheap even unbounded --
        # the big features are paged in per-window by the OS, shared across workers.
        if ei not in self._cache:
            self._cache[ei] = TrackExample.load(self.paths[ei], mmap=True)
        return self._cache[ei]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ei, start = self._index[i]
        ex = self._example(ei)
        # np.asarray copies just this window out of the memmap into a real array.
        feat = np.asarray(ex.features[:, start:start + self.window])
        x = np.zeros((ex.features.shape[0], self.window), dtype=np.float32)
        x[:, : feat.shape[1]] = feat
        y: dict[str, torch.Tensor] = {}
        for h in self.heads:
            t = ex.targets[h][start:start + self.window]
            padded = np.zeros(self.window, dtype=np.float32)
            padded[: t.shape[0]] = t
            y[h] = torch.from_numpy(padded)
        return torch.from_numpy(x).unsqueeze(0), y  # (1, n_bins, W), {head: (W,)}


def collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack windows into (B, 1, n_bins, W) and each target into (B, W)."""
    xs = torch.stack([x for x, _ in batch])
    heads = batch[0][1].keys()
    ys = {h: torch.stack([y[h] for _, y in batch]) for h in heads}
    return xs, ys
