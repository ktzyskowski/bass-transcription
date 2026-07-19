"""Put a reference and an estimate on the same time grid, then pool across tracks.

Every baseline emits its own grid: pyin hops 512 samples at 22.05 kHz (~23 ms),
PESTO defaults to 10 ms, MDB-bass-synth annotates every 128 samples at 44.1 kHz
(~2.9 ms). Nothing can be compared until they agree.

The reference grid wins. That is the mir_eval convention and it is the right one
-- the reference decides where the question "what pitch is sounding?" is asked,
and the estimate answers wherever it is asked. Resampling the reference down to
a coarse estimate grid would instead let a baseline improve its own score by
predicting less often.

Resampling is voicing-aware (``mir_eval.melody.resample_melody_series``):
interpolating Hz straight through an unvoiced gap would invent a glide between
two unrelated notes, so voicing is carried separately and reapplied after.
"""

from dataclasses import dataclass

import mir_eval
import numpy as np

from stemify.baselines.base import F0Contour


def align(
    ref_times: np.ndarray,
    ref_freqs: np.ndarray,
    est: F0Contour,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``est`` onto the reference grid.

    Returns ``(true, pred)``: two equal-length Hz arrays with 0.0 for unvoiced,
    which is exactly the contract ``Metric.calculate`` expects.
    """
    if est.times.size == 0:
        return ref_freqs, np.zeros_like(ref_freqs)

    est_freqs, est_voicing = mir_eval.melody.resample_melody_series(
        times=est.times,
        frequencies=np.abs(est.freqs),
        voicing=(est.freqs > 0).astype(float),
        times_new=ref_times,
    )
    # resample_melody_series returns a soft voicing weight; anything that is not
    # confidently voiced after interpolation is treated as a gap.
    pred = np.where(est_voicing > 0.5, est_freqs, 0.0)
    return ref_freqs, pred


@dataclass(frozen=True)
class FrameSet:
    """Frames pooled across many tracks, with enough labelling to slice them.

    Pooling is what makes aggregation correct for free: a metric called on the
    whole set is frame-weighted, so a six-minute track counts for more than a
    thirty-second one. Averaging per-track scores instead would silently weight
    every track equally, which is the classic way to get an RPA that no
    individual track achieves.

    ``track`` holds an index into ``track_names`` for each frame, so a per-track
    or per-subset breakdown is just ``calculate`` on a masked view.
    """

    true: np.ndarray
    pred: np.ndarray
    track: np.ndarray  # int index into track_names, one per frame
    track_names: list[str]

    def __len__(self) -> int:
        return int(self.true.size)

    @classmethod
    def from_tracks(cls, parts: list[tuple[str, np.ndarray, np.ndarray]]) -> "FrameSet":
        """Build from ``(name, true, pred)`` triples, one per track."""
        if not parts:
            empty = np.array([], dtype=float)
            return cls(empty, empty, np.array([], dtype=int), [])
        names = [name for name, _, _ in parts]
        return cls(
            true=np.concatenate([t for _, t, _ in parts]),
            pred=np.concatenate([p for _, _, p in parts]),
            track=np.concatenate(
                [np.full(t.size, i, dtype=int) for i, (_, t, _) in enumerate(parts)]
            ),
            track_names=names,
        )

    def where(self, mask: np.ndarray) -> "FrameSet":
        """A view holding only the frames selected by ``mask``."""
        return FrameSet(
            true=self.true[mask],
            pred=self.pred[mask],
            track=self.track[mask],
            track_names=self.track_names,
        )

    def for_track(self, name: str) -> "FrameSet":
        return self.where(self.track == self.track_names.index(name))

    @property
    def voiced(self) -> np.ndarray:
        """Mask of reference-voiced frames -- the ones that carry a pitch question."""
        return self.true > 0
