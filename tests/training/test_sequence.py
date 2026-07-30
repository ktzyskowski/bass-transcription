"""SequenceDataset + padding collate: whole-track items and their batching.

The collate is exercised purely (synthetic examples of different lengths); the
dataset is exercised end to end on a tiny synthetic track (real CQT + pyin, fast),
through a DataLoader, so the cache-and-load path and the item->batch wiring are
covered without Slakh.
"""

import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.training.conditions import CONDITIONS
from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem
from stemify.features import SAMPLE_RATE
from stemify.notes import NoteEvent
from stemify.training.datasets.sequence import (
    SequenceBatch,
    SequenceDataset,
    SequenceExample,
    collate,
)


def _example(n_bins: int, t: int, channels: int = 1) -> SequenceExample:
    times = np.arange(t) * 0.0116
    return SequenceExample(
        # (C, n_bins, T): the channel axis is always present, so a single-channel
        # track is (1, n_bins, T) rather than a bare CQT.
        features=np.random.rand(channels, n_bins, t).astype(np.float32),
        frame_times=times,
        contour=Contour(times=times, freqs=np.full(t, 110.0)),
        notes=[NoteEvent(onset=0.1, offset=0.3, midi_pitch=45.0)],
    )


def test_collate_pads_features_and_keeps_per_track_lists():
    a, b = _example(8, 30), _example(8, 20)
    batch = collate([a, b])
    assert isinstance(batch, SequenceBatch)
    assert batch.features.shape == (2, 1, 8, 30)   # padded to the longer track
    assert batch.lengths == [30, 20]
    assert len(batch) == 2
    # the shorter track's tail is zero-padding, its real frames are intact
    assert batch.features[1, 0, :, 20:].abs().sum() == 0
    assert np.allclose(batch.features[1, :, :, :20].numpy(), b.features)
    assert batch.contours[0] is a.contour and batch.notes[1] == b.notes


def test_collate_keeps_every_input_channel():
    # A mix-conditioned semi-CRF reads two stacked CQTs; the channel axis must survive
    # padding, which is where a hardcoded `1` used to sit.
    a, b = _example(8, 30, channels=2), _example(8, 20, channels=2)
    batch = collate([a, b])
    assert batch.features.shape == (2, 2, 8, 30)
    assert np.allclose(batch.features[0, :, :, :30].numpy(), a.features)


def test_collate_rejects_a_bare_cqt():
    # Without the channel axis the batch shape silently depends on who built the
    # example, and a 2-channel model would meet a 1-channel batch at the first conv.
    import pytest

    bare = _example(8, 30)
    bare.features = bare.features[0]
    with pytest.raises(ValueError, match=r"must be \(C, n_bins, T\)"):
        collate([bare])


# --- a tiny synthetic dataset, so the cache-and-load path runs without Slakh --- #
class _Tone(EvalItem):
    def __init__(self, path, hz=110.0):
        self._path, self._hz = path, hz

    @property
    def name(self):
        return self._path.stem

    def audio(self):
        return self._path

    def reference(self):
        t = np.arange(80) * 0.0116
        return Contour(times=t, freqs=np.full(80, self._hz))

    def reference_notes(self):
        return [NoteEvent(onset=0.1, offset=0.8, midi_pitch=45.0)]


class _ToneSet(Dataset):
    def __init__(self, paths):
        self._items = [_Tone(p) for p in paths]

    @property
    def name(self):
        return "tone-set"

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


def _write_tone(path, hz, dur_s=1.0):
    t = np.linspace(0, dur_s, int(SAMPLE_RATE * dur_s), endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * hz * t), SAMPLE_RATE)


def test_sequence_dataset_caches_and_serves_whole_tracks(tmp_path):
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    _write_tone(paths[0], 110.0)
    _write_tone(paths[1], 146.0)
    ds = SequenceDataset(_ToneSet(paths), Cache(tmp_path / "cache"), CONDITIONS["stem"])
    assert len(ds) == 2
    ex = ds[0]
    assert isinstance(ex, SequenceExample)
    assert ex.features.ndim == 3                            # (C, n_bins, T)
    assert ex.features.shape[-1] == ex.frame_times.shape[0]  # CQT paired with its grid
    assert ex.notes and len(ex.contour) > 0

    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    batch = next(iter(loader))
    assert isinstance(batch, SequenceBatch) and len(batch) == 2
    assert batch.features.shape[0] == 2
