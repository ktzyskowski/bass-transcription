"""Note events -> a frame-level reference contour, gated against the audio's decay.

Shared by every note-native dataset, because the question it answers is a property of
*bass* rather than of any one corpus: MIDI says when a note was released, and a
plucked string keeps sounding well past that. Reading voicing straight off note-off is
therefore ~30% too tight on rendered bass (measured on Slakh), and it charges a
tracker a false alarm for correctly following a dying note.

Extracted when FiloBass arrived and needed the same gate. It was written once inside
``slakh.item``; a second copy is how the two drift, and a drift here is invisible --
both datasets still produce a plausible contour, and the one that is 30% too tight
simply reports every tracker as worse.

``transpose`` is the one thing the callers genuinely disagree about, so it is a
parameter rather than a constant: Slakh's rendered bass sounds an octave below its
MIDI (see ``slakh.item.BASS_TRANSPOSE``), while FiloBass is annotated at sounding
pitch and needs none.
"""

from pathlib import Path

import librosa
import numpy as np

from stemify.music.contour import Contour

# Frame grid for the reference contour. 10 ms is the mir_eval melody convention
# and matches PESTO's default step; unlike MDB-bass-synth's 2.9 ms this is our
# choice, since MIDI is event-based and has no native grid.
REFERENCE_HOP_S = 0.01

# Per-note voicing gate. A note stays voiced from onset until its energy falls
# NOTE_DECAY_DB below that note's *own* attack peak (measured over the first
# ATTACK_WINDOW_S). Relative to the note, not the track: a quiet note gets the
# same proportional decay window as a loud one, where a track-relative floor
# would gate the quiet note out early.
NOTE_DECAY_DB = 40.0
ATTACK_WINDOW_S = 0.05


def notes_to_contour(
    notes: list[tuple[float, float, int]],
    audio_path: str | Path,
    transpose: int = 0,
    hop_s: float = REFERENCE_HOP_S,
    decay_db: float = NOTE_DECAY_DB,
) -> Contour:
    """``(start, end, pitch)`` triples, sorted by onset -> ``Contour``, 0 Hz where silent.

    Each note is voiced across its MIDI span -- which is what keeps onsets exact --
    and then through its decay until its energy falls ``decay_db`` below that note's
    own attack peak, or the next note starts, whichever comes first.

    The gate reads whichever audio the *labels* describe, which for a mix-scored run
    is still the isolated stem: this is ground truth about the bass and does not
    change with what a baseline happens to listen to.

    Overlapping notes collapse to the lowest voice (``np.fmin``). That is a real
    choice rather than a tie-break -- the reference is monophonic by construction, and
    on a bass the lowest sounding voice is the line.
    """
    if not notes:
        return Contour.empty()

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, round(sr * hop_s))
    rms = librosa.feature.rms(y=audio, frame_length=hop * 4, hop_length=hop)[0]
    # The audio grid may run past the last note-off -- that overhang is the decay
    # this gate exists to keep, so times come from the audio.
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)
    attack_frames = max(1, round(ATTACK_WINDOW_S / hop_s))
    onsets = [n[0] for n in notes]

    midi = np.full(times.shape, np.nan)
    for i, (start, end, pitch) in enumerate(notes):
        lo = int(np.searchsorted(times, start))
        # A note owns frames up to the next onset; its decay lives in there.
        nxt = onsets[i + 1] if i + 1 < len(notes) else times[-1] + hop_s
        hi = int(np.searchsorted(times, nxt))
        if hi <= lo:
            continue
        peak_db = energy_db[lo : min(lo + attack_frames, hi)].max()
        off = int(np.searchsorted(times, end))
        active = np.arange(lo, hi) < off
        seg_voiced = active | (energy_db[lo:hi] > peak_db - decay_db)
        vals = np.where(seg_voiced, pitch - transpose, np.nan)
        midi[lo:hi] = np.fmin(midi[lo:hi], vals)

    freqs = np.zeros_like(times)
    sounding = ~np.isnan(midi)
    freqs[sounding] = 440.0 * 2.0 ** ((midi[sounding] - 69.0) / 12.0)
    return Contour(times, freqs)
