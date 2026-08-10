"""Transcribed notes in forms a bass player can check.

There is no ground truth for a song off the internet, so the output of this path is not
a number -- it is somebody's judgement. That makes the artifact's job different from a
metric's: it has to make disagreement *fast to express*. Two forms, for the two ways a
player will actually check.

**Listening** is the fast one and needs no tooling. ``preview_wav`` renders the notes
and mixes them over the original, so a wrong pitch or a missed entry is audible in one
pass rather than found by reading.

**Reading** is the precise one. ``notes_csv`` gives times, pitches and note names, in
order, so a specific note can be pointed at rather than described.
"""

import csv
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf

from stemify.core.features import SAMPLE_RATE, load_audio
from stemify.music.notes import NoteEvent

HARMONICS = (1.0, 0.5, 0.25)
"""Relative amplitude of the fundamental and its first two overtones.

A bass fundamental is 41-100 Hz, which laptop speakers and earbuds do not reproduce --
a pure sine preview is *silent* on most of the hardware anyone will check this on. The
overtones put the pitch in a band that reproduces, and the ear infers the missing
fundamental from them, which is the same trick that makes bass audible on a phone."""

FADE_S = 0.008
"""Ramp on each note's edges. Squarely a click-suppressor: a note starting at full
amplitude on a non-zero crossing is a step, and a track of them reads as percussion."""


def note_name(midi_pitch: float) -> str:
    """``E1``, ``A#2`` -- what a player will call the note."""
    return str(librosa.midi_to_note(float(midi_pitch), unicode=False))


def notes_csv(notes: Sequence[NoteEvent], path: Path) -> Path:
    """One row per note: when, how long, what, and how sure.

    Sorted by onset rather than by confidence. A reviewer follows the song, so time
    order is the order they can check against what they are hearing; confidence is a
    column they can sort on later if they want the doubtful ones first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "onset_s", "offset_s", "duration_s", "midi_pitch", "note",
             "confidence"]
        )
        for k, n in enumerate(sorted(notes, key=lambda n: n.onset)):
            writer.writerow([
                k, f"{n.onset:.3f}", f"{n.offset:.3f}", f"{n.offset - n.onset:.3f}",
                f"{n.midi_pitch:.2f}", note_name(n.midi_pitch),
                f"{n.confidence:.3f}",
            ])
    return path


def render(notes: Sequence[NoteEvent], duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The notes as audio: a decaying harmonic tone per note, summed."""
    out = np.zeros(int(duration_s * sr) + 1, dtype=np.float32)
    fade = max(1, int(FADE_S * sr))
    for n in notes:
        start, stop = int(n.onset * sr), min(int(n.offset * sr), out.size)
        if stop <= start:
            continue
        t = np.arange(stop - start, dtype=np.float32) / sr
        freq = float(librosa.midi_to_hz(n.midi_pitch))
        tone = np.zeros_like(t)
        for k, amp in enumerate(HARMONICS):
            tone += amp * np.sin(2 * np.pi * freq * (k + 1) * t)
        # Decay, so overlapping notes stay separable by ear rather than blurring into
        # one sustained pad -- and so a *held* wrong note is audibly held.
        tone *= np.exp(-3.0 * t).astype(np.float32)
        ramp = np.minimum(np.arange(tone.size) / fade, 1.0).astype(np.float32)
        out[start:stop] += tone * ramp * ramp[::-1]
    return out


def preview_wav(
    notes: Sequence[NoteEvent],
    path: Path,
    source: str | Path | None = None,
    sr: int = SAMPLE_RATE,
    source_gain: float = 0.6,
) -> Path:
    """The notes rendered, mixed over ``source`` when given.

    Mixed rather than written alone, because the question a player is answering is
    "does this line up with what I hear", and two files in two players cannot answer it.
    Against the original, a late onset or a wrong pitch beats audibly. Pass
    ``source=None`` for the transcription in isolation, which is the better way to hear
    what the model thinks *without* the original masking it.
    """
    synth = render(notes, max((n.offset for n in notes), default=0.0) + 1.0, sr)
    if source is not None:
        original = load_audio(source, sr=sr).astype(np.float32)
        width = max(original.size, synth.size)
        mixed = np.zeros(width, dtype=np.float32)
        mixed[: original.size] += source_gain * original
        mixed[: synth.size] += synth
        synth = mixed
    peak = float(np.abs(synth).max())
    if peak > 0:
        synth = synth / peak * 0.9        # headroom, so nothing clips on playback
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, synth, sr)
    return path
