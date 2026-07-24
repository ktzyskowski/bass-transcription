"""Frame-grid utilities: nearest-neighbour resampling between grids.

Pure numpy, no model or training deps -- a top-level leaf so the label builder,
the decoders, the segmenters and the trainer can all use these without an import
cycle (it used to live under ``decoders/``, but ``training/labels.py`` needs it
too and shouldn't depend on ``decoders``). Nearest-neighbour is the honest choice
for a mask or an onset: interpolation would invent half-voiced frames or smear an
onset across two.
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
