"""BassStem: one Slakh bass stem as an EvalItem, and the constants it needs.

Split out of a 453-line slakh.py. What one item *is* -- its audio roles, its
reference contour, its notes -- changes for different reasons than how the corpus tree
is walked (index) or how the collection is filtered (dataset). See the package
docstring for the corpus-level facts, including why bass MIDI is transposed.
"""

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.data.base import EvalItem
from stemify.notes import NoteEvent
from stemify.sources import Sources

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

# Slakh's rendered bass audio sounds one octave BELOW its MIDI note numbers.
# Electric bass is a transposing instrument -- written an octave above where it
# sounds -- and the sample libraries Slakh rendered with follow that convention.
#
# Measured, not assumed: pyin on clean Slakh bass stems disagreed with a
# naive MIDI-to-Hz reference by exactly -12 semitones on 95.8% of frames, giving
# RPA 0.7% against RCA 93.9%. Confirmed across all three bass programs present
# (Acoustic Bass, Electric Bass (finger), Fretless Bass) at 96-98% of frames
# over 26 tracks. The corrected distribution also lines up with reality: median
# MIDI 41 against MDB-bass-synth's 40, where the uncorrected figure was 53 and
# contained nothing at all below MIDI 47 -- i.e. nothing in the bottom 19
# semitones of a bass's range, which is precisely where the problem is hard.
#
# Without this every Slakh label is an octave high. A model trained on them
# would learn to predict an octave sharp and still look correct on Slakh's own
# evaluation, because the error is consistent -- and fail everywhere else.
BASS_TRANSPOSE = 12


@dataclass(frozen=True)
class BassStem(EvalItem):
    """One rendered bass stem, its MIDI, and the mix it sits inside."""

    track: str  # e.g. "Track00042"
    split: str
    stem_id: str  # e.g. "S03"
    stem_path: Path
    midi_path: Path
    mix_path: Path
    program_name: str
    n_bass_stems: int  # how many bass stems the parent track has
    source: str = "stem"

    @property
    def name(self) -> str:
        return f"{self.track}/{self.stem_id}"

    @property
    def groups(self) -> dict[str, str]:
        return {
            "split": self.split,
            "source": self.source,
            "program": self.program_name,
        }

    @property
    def unambiguous(self) -> bool:
        """True when this is the track's only bass.

        Mix-based evaluation is only meaningful for these: with two basses
        sounding, a single reference contour cannot describe what is in the mix,
        so a model is being scored against one of two right answers.
        """
        return self.n_bass_stems == 1

    def audio(self) -> Path:
        return self.stem_path if self.source == "stem" else self.mix_path

    def sources(self) -> Sources:
        """The audio ``source`` selected, plus the mixture it sits inside.

        ``mix`` is present unconditionally: Slakh ships ``mix.flac`` beside every
        stem, so a mix-conditioned channel costs nothing to feed and needs no
        re-rendering. Note this is the *raw* mixture -- a condition that wants it
        separated says so with its own transform, which is what keeps the
        demucs-bass and raw-mix channels on distinct cache keys for the same file.
        """
        return Sources(bass=self.audio(), mix=self.mix_path)

    @cached_property
    def _notes(self) -> list[tuple[float, float, int]]:
        """(start, end, pitch) for every note, sorted by onset."""
        import pretty_midi  # deferred: pulls in mido at import time

        with warnings.catch_warnings():
            # Lakh MIDI is full of harmless oddities (stray key signatures,
            # out-of-range tempo events); they are not our problem to fix.
            warnings.simplefilter("ignore")
            midi = pretty_midi.PrettyMIDI(str(self.midi_path))
        notes = [
            (n.start, n.end, n.pitch)
            for inst in midi.instruments
            if not inst.is_drum
            for n in inst.notes
        ]
        return sorted(notes)

    def reference(
        self, hop_s: float = REFERENCE_HOP_S, decay_db: float = NOTE_DECAY_DB
    ) -> Contour:
        """Frame-level ``(times, freqs)``, 0 Hz where no bass sounds.

        Voicing is gated **per note against the note's own attack**, not read
        straight from MIDI note-off. A rendered bass note keeps sounding through
        its decay after MIDI says it ended -- measured, ~31% of MIDI-silent
        frames still carry audible bass -- so pure note-on/off voicing is ~30%
        too tight and charges a tracker a false alarm for correctly following a
        dying note. Each note is voiced across its MIDI span and then through its
        decay until its energy falls ``decay_db`` below that note's own attack
        peak (or the next note starts). Per-note rather than track-relative so a
        quiet note gets the same proportional decay window as a loud one.

        The gate reads the *stem* even when the mix will be scored, because this
        is ground truth about the bass, independent of what a baseline listens
        to. Overlapping notes collapse to the lowest voice (``np.fmin``).
        """
        notes = self._notes
        if not notes:
            return Contour.empty()

        audio, sr = librosa.load(self.stem_path, sr=None, mono=True)
        hop = max(1, round(sr * hop_s))
        rms = librosa.feature.rms(y=audio, frame_length=hop * 4, hop_length=hop)[0]
        # The audio grid may run past the last note-off -- that overhang is the
        # decay this gate exists to keep, so times come from the audio.
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
            # Voiced across the MIDI span (precise onsets), then through decay
            # while energy stays within decay_db of this note's own attack.
            active = np.arange(lo, hi) < off
            seg_voiced = active | (energy_db[lo:hi] > peak_db - decay_db)
            vals = np.where(seg_voiced, pitch - BASS_TRANSPOSE, np.nan)
            midi[lo:hi] = np.fmin(midi[lo:hi], vals)

        freqs = np.zeros_like(times)
        sounding = ~np.isnan(midi)
        freqs[sounding] = 440.0 * 2.0 ** ((midi[sounding] - 69.0) / 12.0)
        return Contour(times, freqs)

    def reference_notes(self) -> list[NoteEvent]:
        """The MIDI notes as events, octave-corrected like ``reference``.

        MIDI is note-native, so this is the truth in its original form rather
        than a segmentation of a contour -- which is exactly why Slakh, not
        MDB-bass-synth, is the dataset for note-level evaluation.
        """
        return [
            NoteEvent(onset=start, offset=end, midi_pitch=pitch - BASS_TRANSPOSE)
            for start, end, pitch in self._notes
        ]

    def polyphony_rate(self, hop_s: float = REFERENCE_HOP_S) -> float:
        """Fraction of sounding frames with more than one note active.

        The monophonic assumption is load-bearing for this project, so this is
        the number that says whether a track honours it. Octave doubling in the
        source transcription is the usual cause.
        """
        notes = self._notes
        if not notes:
            return 0.0
        end = max(e for _, e, _ in notes)
        times = np.arange(0.0, end + hop_s, hop_s)
        active = np.zeros(times.shape, dtype=int)
        for start, stop, _ in notes:
            lo, hi = np.searchsorted(times, [start, stop])
            active[lo:hi] += 1
        sounding = active > 0
        if not sounding.any():
            return 0.0
        return float((active[sounding] > 1).mean())

