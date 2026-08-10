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

from collections.abc import Iterable, Sequence

import numpy as np

from stemify.music.contour import Contour
from stemify.music.grids import nearest_frame, resample_mask
from stemify.music.notes import NoteEvent


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
    centres = nearest_frame(frame_times, np.asarray(onset_times, dtype=float))
    for c in centres:
        lo = max(0, int(c) - tolerance_frames)
        hi = min(frame_times.size, int(c) + tolerance_frames + 1)
        y[lo:hi] = 1.0
    return y


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
    nearest = nearest_frame(pitch_times[voiced], frame_times)
    resampled = src_midi[voiced][nearest]
    # A frame closer to a rest than to any voiced source frame stays NaN.
    src_voiced_time = pitch_times[voiced]
    gap = np.abs(src_voiced_time[nearest] - frame_times)
    hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.0
    y[gap <= hop] = resampled[gap <= hop]
    return y


def _hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Frequency to MIDI semitones: ``69 + 12 log2(f / 440)``.

    Written out rather than taken from librosa, which is the only reason this pillar
    would need a third-party audio library at all. It is the definition of the MIDI
    scale, not a signal-processing choice, so it belongs with the notes.
    """
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def build_targets(
    frame_times: np.ndarray,
    reference: Contour,
    notes: Sequence[NoteEvent],
    heads: Iterable[str],
    onset_tolerance_frames: int = 1,
) -> dict[str, np.ndarray]:
    """The target each requested head needs, all on ``frame_times``.

    One entry point rather than three calls, because *which* target a head takes is a
    property of the head name and every caller was rediscovering it. Requesting a head
    nothing here builds yields no entry rather than an error -- the caller's cache then
    reports it missing, which is where a wrong head name is actually visible.

    ``reference`` is the *gated* reference contour (voicing and pitch); ``notes`` give
    the exact onsets. Labels always come from the clean stem's truth no matter which
    audio the features were computed from, which is what makes a separated-audio run
    comparable to a clean one.

    Pure: arrays in, arrays out, no disk and no corpus. Wrapping the result in a
    ``CachedItem`` is the persistence layer's business, and used to be welded in here.
    """
    heads = set(heads)
    targets: dict[str, np.ndarray] = {}
    if "onset" in heads:
        onsets = np.array([n.onset for n in notes], dtype=float)
        targets["onset"] = onset_targets(onsets, frame_times, onset_tolerance_frames)
    if "voicing" in heads:
        targets["voicing"] = voicing_targets(
            reference.times, reference.voiced, frame_times
        )
    if "pitch" in heads:
        midi = np.full(reference.freqs.shape, np.nan, dtype=float)
        voiced = reference.freqs > 0
        midi[voiced] = _hz_to_midi(reference.freqs[voiced])
        targets["pitch"] = pitch_targets(reference.times, midi, frame_times)
    return targets
