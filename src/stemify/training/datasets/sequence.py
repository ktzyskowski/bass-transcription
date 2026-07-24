"""Whole-track sequence data for structured (semi-CRF) training.

The frame-head path serves fixed windows (``dataset.py``); the structured decoder
needs *whole* tracks -- a note can exceed any window, and the segmental DP scans the
entire sequence. ``SequenceDataset`` yields one track per item (its CQT, grid, pyin
contour and reference notes); the padding ``collate`` stacks a minibatch to a common
length, which is what lets the batched forward algorithm run one scan across the
batch. Both the CQT and the contour are cached on disk exactly as the frame-head
trainer and the async validator cache them, so a stem and a demucs pass of a track
reuse the same tensors and never recompute.

This pairs with ``SemiCRFLoss`` the way ``FrameWindowDataset``/``collate`` pairs with
``FrameBCELoss`` -- the batch shape and the objective are the two things that vary
between training paradigms; everything else is the shared ``Trainer``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.notes import NoteEvent
from stemify.pitch import PyinPitch
from stemify.serde import TrackExample
from stemify.training.cache import cache_features, cached_f0, example_path
from stemify.transforms import AudioTransform, Identity


@dataclass
class SequenceExample:
    """One whole track: the CQT, its frame grid, the pyin contour and reference
    notes. The item a ``SequenceDataset`` yields; ``collate`` stacks a batch of them.
    A named bundle rather than a tuple because two fields are objects (a ``Contour``,
    a note list), not stackable arrays."""

    features: np.ndarray      # (n_bins, T) CQT
    frame_times: np.ndarray   # (T,)
    contour: Contour          # pyin pitch track (its own grid)
    notes: list[NoteEvent]    # reference notes -> gold segmentation


@dataclass
class SequenceBatch:
    """A minibatch of tracks: the CQT padded to a common length, the rest per-track
    lists. Only the CQT is a rectangle to stack; a contour lives on the pyin grid and
    a note set is variable-length, so neither pads. ``lengths`` rides along so the
    model output and the DP are read at each track's true end."""

    features: torch.Tensor           # (B, 1, n_bins, Tmax), zero-padded on the right
    lengths: list[int]               # true frame count per track
    frame_times: list[np.ndarray]
    contours: list[Contour]
    notes: list[list[NoteEvent]]

    def __len__(self) -> int:
        return len(self.lengths)


def collate(batch: Sequence[SequenceExample]) -> SequenceBatch:
    """Pad a minibatch of whole tracks to a common length along time (zeros right)."""
    lengths = [ex.features.shape[1] for ex in batch]
    n_bins, tmax = batch[0].features.shape[0], max(lengths)
    features = torch.zeros(len(batch), 1, n_bins, tmax)
    for b, ex in enumerate(batch):
        features[b, 0, :, : lengths[b]] = torch.from_numpy(np.ascontiguousarray(ex.features))
    return SequenceBatch(
        features=features,
        lengths=lengths,
        frame_times=[ex.frame_times for ex in batch],
        contours=[ex.contour for ex in batch],
        notes=[ex.notes for ex in batch],
    )


class SequenceDataset(Dataset):
    """Whole-track examples for a dataset, caching the CQT + pyin contour once.

    Mirrors the async validator's prep (``cache_features`` for the CQT, ``cached_f0``
    for the contour) minus the per-frame targets -- the structured loss reads notes,
    not frame labels. Labels always come from the clean stem's notes, whatever audio
    the features were drawn from. Items hold only the feature *path* (loaded per
    ``__getitem__``) plus the small contour/notes, so a large set never sits in RAM
    and the dataset pickles cheaply across DataLoader workers."""

    def __init__(
        self,
        dataset: SlakhLikeDataset,
        cache_dir: Path,
        transform: AudioTransform | None = None,
        contour_cache: Path | None = None,
    ) -> None:
        transform = transform or Identity()
        estimator = PyinPitch()
        cache_features(dataset, cache_dir, transform)
        self._items: list[tuple[Path, Contour, list[NoteEvent]]] = []
        for item in dataset:
            fpath = example_path(cache_dir, item.name, transform.name)
            audio = transform.apply(item.audio())
            contour = cached_f0(estimator, audio, contour_cache)
            self._items.append((fpath, contour, item.reference_notes()))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> SequenceExample:
        fpath, contour, notes = self._items[i]
        track = TrackExample.load(fpath, mmap=True)
        return SequenceExample(
            features=np.asarray(track.features),  # materialise this track's CQT
            frame_times=track.frame_times,
            contour=contour,
            notes=notes,
        )
