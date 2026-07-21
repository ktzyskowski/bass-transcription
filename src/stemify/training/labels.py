"""Per-frame training targets, on the CQT frame grid.

Each head needs a different target from the same track, but all on the *same*
grid as the features -- the CQT frame times (~11.6 ms), not the 10 ms reference
grid the metrics use. These builders take explicit arrays (onset times, a source
voicing mask, per-frame pitch) rather than a dataset object, so they are pure and
testable with hand-built inputs; ``dataset.py`` adapts a Slakh item to them.

The grid mismatch is why every builder resamples to ``frame_times`` by nearest
neighbour -- the honest choice for a mask or an onset, where interpolation would
invent half-voiced frames or smear an onset across two.
"""

import numpy as np


def _nearest_frame(frame_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest frame to each query time. Vectorised, so a whole
    track's onsets or a whole source grid resample in one call."""
    j = np.searchsorted(frame_times, query)
    j = np.clip(j, 0, frame_times.size - 1)
    left = np.clip(j - 1, 0, frame_times.size - 1)
    choose_left = np.abs(frame_times[left] - query) < np.abs(frame_times[j] - query)
    return np.where(choose_left, left, j)


def onset_targets(
    onset_times: np.ndarray,
    frame_times: np.ndarray,
    tolerance_frames: int = 1,
) -> np.ndarray:
    """1.0 at frames on (or within ``tolerance_frames`` of) a note onset, else 0.

    A single hard frame is too sparse to learn -- a prediction one frame early is
    as useful as an exact hit but would score as a miss -- so the target is
    widened by ``tolerance_frames`` on each side, the frame analogue of the 50 ms
    onset tolerance the note metric already allows. Peak-picking at inference
    collapses the plateau back to one onset.
    """
    y = np.zeros(frame_times.size, dtype=np.float32)
    if len(onset_times) == 0 or frame_times.size == 0:
        return y
    centres = _nearest_frame(frame_times, np.asarray(onset_times, dtype=float))
    for c in centres:
        lo = max(0, int(c) - tolerance_frames)
        hi = min(frame_times.size, int(c) + tolerance_frames + 1)
        y[lo:hi] = 1.0
    return y


def resample_mask(
    src_times: np.ndarray,
    src_mask: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Resample a boolean mask from its own grid onto ``frame_times`` (nearest)."""
    y = np.zeros(frame_times.size, dtype=np.float32)
    if src_times.size == 0 or frame_times.size == 0:
        return y
    nearest = _nearest_frame(np.asarray(src_times, dtype=float), frame_times)
    return np.asarray(src_mask)[nearest].astype(np.float32)


def voicing_targets(
    voiced_times: np.ndarray,
    voiced_mask: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Per-frame voiced/unvoiced target, resampled onto the feature grid.

    The source mask is the *gated* reference voicing (decay-aware), not raw MIDI
    note spans -- the model's voicing ceiling is this mask's fidelity, which is
    why the gate must be validated against annotated voicing before this head is
    trusted (see the frame-level findings).
    """
    return resample_mask(voiced_times, voiced_mask, frame_times)


def pitch_targets(
    pitch_times: np.ndarray,
    pitch_midi: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Per-frame MIDI pitch on the feature grid; NaN where unvoiced.

    Left continuous (not yet binned) because the output encoding is still open --
    plain classification over CQT bins, or the two-hot cents target discussed for
    multimodality. Binning is the head's concern; this just carries the truth.
    """
    y = np.full(frame_times.size, np.nan, dtype=np.float32)
    if pitch_times.size == 0 or frame_times.size == 0:
        return y
    src_midi = np.asarray(pitch_midi, dtype=float)
    voiced = ~np.isnan(src_midi)
    if not voiced.any():
        return y
    nearest = _nearest_frame(pitch_times[voiced], frame_times)
    resampled = src_midi[voiced][nearest]
    # A frame closer to a rest than to any voiced source frame stays NaN.
    src_voiced_time = pitch_times[voiced]
    gap = np.abs(src_voiced_time[nearest] - frame_times)
    hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.0
    y[gap <= hop] = resampled[gap <= hop]
    return y