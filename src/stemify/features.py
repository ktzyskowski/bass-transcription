"""Audio to model input: loading, and the CQT front-end.

This lives outside ``data/`` on purpose. A dataset loader answers "what
recordings exist and what is true of them"; this answers "how does a waveform
become something a model can consume", which is the same regardless of which
dataset the waveform came from.

The CQT is log-frequency spaced, so a pitch change is a fixed shift along the
frequency axis regardless of register -- which is what lets a convolutional
model share weights across pitches.

The constants encode decisions that would otherwise have to be re-derived (why 3
bins per semitone, why 6 octaves starting a semitone below the open low E): the
reasoning is the expensive part, not the code. ``midi_to_bin`` has no caller yet
(it is for a future pitch head that regresses onto the CQT grid), kept for the
same reason.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050

# Bass fundamentals run from E1 (open low E, ~41 Hz) to G4 (~392 Hz, string 4
# fret 24). Start a semitone below E1 for margin and span 6 octaves at 3 bins
# per semitone, which reaches ~2.6 kHz and so covers the first few harmonics
# too -- a low-E fundamental is weak on many recordings and the harmonic stack
# is what actually disambiguates it.
FMIN = librosa.note_to_hz("D#1")
BINS_PER_SEMITONE = 3
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = 6 * BINS_PER_OCTAVE
HOP_LENGTH = 256  # ~11.6 ms per frame at 22.05 kHz
# NB there is a *second* ``HOP_LENGTH`` in ``pitch.pyin``, at 512. They are different
# grids, not a duplicate: this one is the CQT's, that one is pyin's, and pyin's is
# twice as coarse. The name collision is why the mismatch stayed invisible for so long
# -- ``group_contour`` works on whichever grid the contour arrived on, so a note
# boundary from a pyin contour is quantised to 23.2 ms against a 50 ms match tolerance.
# That interaction cost 7.5 strict F1 (see decoders/heuristic.PITCH_SPLIT_SEMITONES),
# and the learned pitch head removes it by landing the contour on *this* grid.
TOP_DB = 80.0     # dB floor for the log-magnitude conversion

# Identity of the CQT *as configured here* -- the producer half of a cached feature's
# key (see ``stemify.cache``). It names every constant above that changes the output,
# so editing one invalidates the cache instead of silently serving features computed
# under the old settings to a model trained on the new ones. That failure is the quiet
# kind: a stale CQT is still a valid-looking array of the right shape.
SIGNATURE = (
    f"cqt-sr{SAMPLE_RATE}-fmin{FMIN:.4f}-nbins{N_BINS}"
    f"-bpo{BINS_PER_OCTAVE}-hop{HOP_LENGTH}-topdb{TOP_DB:g}"
)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a 24-bit WAV as mono float32, resampled to ``sr``."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def cqt(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a log-magnitude CQT, shaped (n_bins, n_frames).

    Output is in dB relative to the clip's own peak and clipped to an 80 dB
    floor, which normalises away the level differences between pickup settings.
    """
    spec = np.abs(
        librosa.cqt(
            audio,
            sr=sr,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
    )
    return librosa.amplitude_to_db(spec, ref=np.max, top_db=80.0)


def midi_to_bin(midi_pitch: float) -> float:
    """Map a MIDI pitch to its (fractional) CQT bin index."""
    hz = librosa.midi_to_hz(midi_pitch)
    return BINS_PER_OCTAVE * np.log2(hz / FMIN)


def frame_times_for(features: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Centre time of each CQT frame -- the grid every per-frame target is built on
    and every other signal (onsets, the contour) resamples onto. A pure function of
    the frame count and the hop, so it lives with the CQT that defines that hop."""
    return librosa.frames_to_time(
        np.arange(features.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )


@dataclass(frozen=True)
class FrameFeatures:
    """CQT features and the frame-time grid they live on.

    Kept together because a slice of features is meaningless without the times its
    frames fall on, and pairing them as one object means the two can't be handed
    over for different tracks by accident -- the same reason ``Contour`` bundles
    its arrays. ``values`` is the model's input; the pyin ``Contour`` is
    deliberately *not* here (different producer, cache, grid and consumer).

    ``values`` is ``(n_bins, T)`` for the usual single-input model, or
    ``(n_channels, n_bins, T)`` when several renderings of the same music are fed
    side by side (bass in channel 0, the mixture it came from in channel 1). The
    channel axis leads so that a slice over time is the last axis either way, and
    so the array drops straight into a conv front-end's channel dimension."""

    values: np.ndarray        # (n_bins, T) or (C, n_bins, T) -- the model's input
    frame_times: np.ndarray   # (T,)


def stack_channels(channels: Sequence[FrameFeatures]) -> FrameFeatures:
    """Several renderings of one track -> one ``(C, n_bins, T)`` model input.

    Truncates to the shortest channel. Separate files for the same music can differ
    by a frame or two -- a separator's output length, a stem trimmed at render --
    and since every channel is a CQT on the *same* hop, frame *k* is the same
    instant in all of them. So the overhang is surplus at the tail, not a
    misalignment, and dropping it aligns them exactly with nothing to resample.
    (Measured on Slakh: stem, mix and demucs-bass agree to the frame on every track
    checked, so this is a guard, not a routine correction.)

    A single channel passes through untouched, keeping ``(n_bins, T)`` -- the
    one-channel path never pays for a copy or gains a stray axis.
    """
    if not channels:
        raise ValueError("stack_channels needs at least one channel")
    if len(channels) == 1:
        return channels[0]
    n = min(c.values.shape[-1] for c in channels)
    return FrameFeatures(
        np.stack([c.values[..., :n] for c in channels]), channels[0].frame_times[:n]
    )


def extract_features(path: str | Path, sr: int = SAMPLE_RATE) -> FrameFeatures:
    """Audio file -> CQT features with the frame grid they sit on.

    The audio->model-input step every caller shares (training prep, the segmenter,
    validation), in one place so the CQT is never recomputed with drifting params.
    Takes a path (not audio) because callers pass a transform-resolved file."""
    values = cqt(load_audio(path, sr=sr), sr=sr)
    return FrameFeatures(values, frame_times_for(values, sr=sr))
