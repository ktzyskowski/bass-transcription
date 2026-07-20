"""Audio to model input: loading, and the CQT front-end.

This lives outside ``data/`` on purpose. A dataset loader answers "what
recordings exist and what is true of them"; this answers "how does a waveform
become something a model can consume", which is the same regardless of which
dataset the waveform came from.

The CQT is log-frequency spaced, so a pitch change is a fixed shift along the
frequency axis regardless of register -- which is what lets a convolutional
model share weights across pitches.

**Status:** ``load_audio`` is used by the pyin baseline. ``cqt`` and
``midi_to_bin`` have no callers yet -- they are the front-end for the CNN that
does not exist. They are kept rather than deleted because the constants encode
decisions that would otherwise have to be re-derived (why 3 bins per semitone,
why 6 octaves starting a semitone below the open low E), and that reasoning is
the expensive part, not the code.
"""

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
