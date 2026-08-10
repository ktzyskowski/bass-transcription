"""Item: one Slakh bass stem as an AudioItem, and the constants it needs.

Split out of a 453-line slakh.py. What one item *is* -- its audio roles, its
reference contour, its notes -- changes for different reasons than how the corpus tree
is walked (index) or how the collection is filtered (dataset). See the package
docstring for the corpus-level facts, including why bass MIDI is transposed.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from stemify.music.contour import Contour
from stemify.corpus.midi import Note as MidiNote, read_notes
from stemify.corpus.midi_contour import (
    NOTE_DECAY_DB, REFERENCE_HOP_S, notes_to_contour,
)
from stemify.music.notes import NoteEvent
from stemify.audio.sources import Sources

# ``REFERENCE_HOP_S`` and ``NOTE_DECAY_DB`` are imported rather than defined here: the
# grid and the decay gate are facts about *bass*, not about Slakh, and FiloBass needs
# the same ones. They stay importable from this module because ``slakh/__init__``
# re-exports ``REFERENCE_HOP_S`` and callers outside the package read it there.

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
class Item:
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

    def _audio(self) -> Path:
        """Which rendering this item is *about*, per its configured ``source``.
        Private: the contract is ``sources()``, and this is how one of its roles is
        chosen."""
        return self.stem_path if self.source == "stem" else self.mix_path

    def sources(self) -> Sources:
        """The audio ``source`` selected, plus the mixture it sits inside.

        ``mix`` is present unconditionally: Slakh ships ``mix.flac`` beside every
        stem, so a mix-conditioned channel costs nothing to feed and needs no
        re-rendering. Note this is the *raw* mixture -- a condition that wants it
        separated says so with its own transform, which is what keeps the
        demucs-bass and raw-mix channels on distinct cache keys for the same file.
        """
        return Sources(bass=self._audio(), mix=self.mix_path)

    @cached_property
    def _notes(self) -> list[MidiNote]:
        """``(start, end, pitch)`` for every note, sorted by onset -- at *written*
        pitch. ``BASS_TRANSPOSE`` is applied by the two readers below, not here, so the
        raw file's numbers stay inspectable."""
        return read_notes(self.midi_path)

    def reference_contour(
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

        The gate itself lives in ``corpus.midi_contour``: it is a fact about bass
        decay rather than about Slakh, and FiloBass reads the same one. What stays
        here is the only part that is Slakh's own -- ``BASS_TRANSPOSE``.
        """
        return notes_to_contour(
            self._notes, self.stem_path, BASS_TRANSPOSE, hop_s, decay_db
        )

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

