"""Loader for Slakh2100: 1710 synthesized multitracks with aligned MIDI.

The scale dataset, and the only one on the roadmap that is **CC BY 4.0** --
commercially clean, unlike IDMT-SMT-Bass and MDB-bass-synth. Audio was rendered
from MIDI with 187 sampled instrument patches, so the MIDI describes the audio
exactly and labels are free.

What it uniquely provides:

  - **Note-level ground truth.** MIDI carries onsets and offsets, not just a
    contour, so Slakh is what eventually unlocks note segmentation and
    note-level P/R/F. MDB-bass-synth can never do that.
  - **Mix and isolated stem for identical material**, which turns "does the bass
    survive an arrangement" and "does source separation earn its place" into
    controlled experiments rather than guesses.

Layout on disk::

    <root>/<split>/TrackXXXXX/
        metadata.yaml       per-stem instrument class, program, render flags
        mix.flac            the full mix
        stems/SNN.flac      one per rendered source
        MIDI/SNN.mid        the MIDI that produced that stem
        all_src.mid

Splits are ``train``/``validation``/``test``, plus ``omitted`` holding tracks
whose MIDI duplicates another track's -- the redux release moves them aside so
the same material cannot land on both sides of a split. ``index`` skips
``omitted`` unless asked for it explicitly.

Two things differ from the annotated datasets and are worth knowing:

**The reference grid is a choice, not a given.** MDB-bass-synth ships a 2.9 ms
annotation grid; MIDI has no grid at all, so ``REFERENCE_HOP_S`` is ours to set.

**Bass MIDI is not guaranteed monophonic.** Slakh's MIDI comes from Lakh, i.e.
real transcriptions, which contain octave doubling and overlapping releases. The
project assumes one bass note at a time (see docs/pitch-primer.md section 5), so
``reference`` resolves overlaps by taking the lowest sounding pitch and
``polyphony_rate`` reports how often it had to -- a track with a high rate
violates the premise and should probably be filtered out rather than trusted.
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem
from stemify.notes import NoteEvent

# Frame grid for the reference contour. 10 ms is the mir_eval melody convention
# and matches PESTO's default step; unlike MDB-bass-synth's 2.9 ms this is our
# choice, since MIDI is event-based and has no native grid.
REFERENCE_HOP_S = 0.01

# Slakh's instrument taxonomy labels bass stems with this inst_class.
BASS_CLASS = "Bass"

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

SPLITS = ("train", "validation", "test")


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

    def reference(self, hop_s: float = REFERENCE_HOP_S) -> Contour:
        """Frame-level ``(times, freqs)``, 0 Hz where no note sounds.

        Overlapping notes collapse to the lowest sounding pitch -- this is a
        bass line, and the lowest voice is the one a bass tracker should follow.
        See ``polyphony_rate`` for how often that collapse is doing real work.
        """
        notes = self._notes
        if not notes:
            return Contour.empty()

        end = max(e for _, e, _ in notes)
        times = np.arange(0.0, end + hop_s, hop_s)
        # MIDI pitch per frame; NaN reads as silence and keeps np.fmin simple.
        pitches = np.full(times.shape, np.nan)
        for start, stop, pitch in notes:
            lo, hi = np.searchsorted(times, [start, stop])
            pitches[lo:hi] = np.fmin(pitches[lo:hi], pitch)

        sounding = ~np.isnan(pitches)
        freqs = np.zeros_like(times)
        # 440 * 2^((p - 69) / 12), i.e. librosa.midi_to_hz without the import.
        freqs[sounding] = 440.0 * 2.0 ** (
            (pitches[sounding] - BASS_TRANSPOSE - 69.0) / 12.0
        )
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


# The tarball unpacks into a directory of its own, so extracting it into
# data/raw/slakh2100 yields data/raw/slakh2100/slakh2100_flac_redux/<split>/...
# Everyone gets this wrong once; descend automatically rather than making the
# configured root depend on how the archive happened to be unpacked.
ARCHIVE_DIR = "slakh2100_flac_redux"


def resolve_root(root: str | Path) -> Path:
    """The directory that actually holds the split folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Slakh2100 is a single 104 GB tarball from "
            "https://zenodo.org/records/4599666 -- see docs/datasets.md."
        )
    nested = root / ARCHIVE_DIR
    if nested.is_dir() and not any((root / s).is_dir() for s in SPLITS):
        return nested
    return root


def read_metadata(track_dir: Path) -> dict:
    """A track's ``metadata.yaml`` stem table, or ``{}`` if unreadable.

    Returning empty on failure is deliberate: callers treat "no metadata" as
    "this track has no identifiable bass", which is the safe reading for both
    indexing (skip it) and pruning (delete nothing).
    """
    meta_path = track_dir / "metadata.yaml"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open() as fh:
            meta = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return {}
    return meta.get("stems") or {}


def bass_stem_ids(track_dir: Path, require_rendered: bool = True) -> set[str]:
    """Stem ids in ``track_dir`` that hold bass.

    The single definition of "which stems are bass", used both by ``index`` and
    by the pruning tool. These previously disagreed -- pruning matched on
    ``inst_class`` alone while indexing also required the render flags -- which
    meant a change to one silently failed to reach the other.

    ``require_rendered`` is what separates the two callers. Indexing wants only
    stems with audio on disk. Pruning wants every stem the dataset *calls* bass,
    including ones that were never rendered, so that a metadata quirk can never
    cause real bass audio to be deleted.
    """
    stems = read_metadata(track_dir)
    ids = set()
    for stem_id, info in stems.items():
        if info.get("inst_class") != BASS_CLASS or info.get("is_drum"):
            continue
        # Not every stem in the metadata was actually rendered or saved; the
        # flags are the dataset's own record of that, and trusting them avoids
        # a pile of missing-file errors deep in a training run.
        if require_rendered and (
            info.get("audio_rendered") is False or info.get("midi_saved") is False
        ):
            continue
        ids.add(stem_id)
    return ids


def _bass_stems(track_dir: Path, split: str) -> list[BassStem]:
    stems = read_metadata(track_dir)
    mix = track_dir / "mix.flac"
    found = []
    for stem_id in sorted(bass_stem_ids(track_dir)):
        stem_path = track_dir / "stems" / f"{stem_id}.flac"
        midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        # Only the MIDI is universally required -- it is the reference, so
        # without it there is nothing to score against. The stem audio is
        # needed for source="stem" and never read for source="mix", so
        # requiring it here would discard tracks that are perfectly usable for
        # mix evaluation. SlakhDataset applies the source-specific check.
        if not midi_path.exists():
            continue
        program = stems.get(stem_id, {}).get("midi_program_name", "")
        found.append((stem_id, stem_path, midi_path, program))

    return [
        BassStem(
            track=track_dir.name,
            split=split,
            stem_id=stem_id,
            stem_path=stem_path,
            midi_path=midi_path,
            mix_path=mix,
            program_name=program,
            n_bass_stems=len(found),
        )
        for stem_id, stem_path, midi_path, program in found
    ]


def index(
    root: str | Path,
    splits: tuple[str, ...] = SPLITS,
    unambiguous_only: bool = False,
) -> list[BassStem]:
    """Index every rendered bass stem under ``root``.

    A track with two bass stems yields two entries. Pass ``unambiguous_only`` to
    keep only single-bass tracks, which is what mix-based evaluation needs.
    """
    root = resolve_root(root)

    stems: list[BassStem] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for track_dir in sorted(split_dir.iterdir()):
            if track_dir.is_dir():
                stems.extend(_bass_stems(track_dir, split))

    if not stems:
        seen = sorted({p.name for p in root.iterdir()}) if root.is_dir() else []
        raise FileNotFoundError(
            f"No bass stems under {root}. Top-level entries: {seen or 'none'}. "
            f"Expected <root>/<split>/TrackXXXXX/metadata.yaml with splits {SPLITS}."
        )
    if unambiguous_only:
        stems = [s for s in stems if s.unambiguous]
    return stems


class SlakhDataset(Dataset):
    """Slakh2100 bass stems, filtered to those the monophonic premise holds for.

    Defaults to the ``test`` split. Slakh is large enough to train on, so
    evaluating on ``train`` reports a number that stops meaning anything the
    moment a model exists.

    Stems whose MIDI is substantially polyphonic are dropped: a doubled-octave
    transcription has no single right answer, and scoring against one would
    manufacture octave errors that are the transcription's fault rather than the
    model's -- especially misleading in a project whose central metric is the
    octave-error rate.

    The polyphony filter reads every candidate's MIDI, so construction is the
    expensive part and iteration is cheap. Pass ``limit`` to cut that cost while
    iterating.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        source: str = "stem",
        limit: int | None = None,
        max_polyphony: float = 0.05,
        unambiguous_only: bool | None = None,
    ) -> None:
        if source not in ("stem", "mix"):
            raise ValueError(f"source must be 'stem' or 'mix', got {source!r}")
        self.split = split
        self.source = source
        # Mixes *require* single-bass tracks -- with two basses sounding, one
        # reference contour cannot describe the mix. Stems do not, so the two
        # sources otherwise yield different track sets (147 vs 124 on test),
        # which would silently make a stem-vs-mix comparison uncontrolled.
        # Pass unambiguous_only=True explicitly to hold the set fixed across
        # conditions.
        if unambiguous_only is None:
            unambiguous_only = source == "mix"
        self.unambiguous_only = unambiguous_only
        stems = index(root, splits=(split,), unambiguous_only=unambiguous_only)
        if limit:
            stems = stems[:limit]

        # Stem audio is only required when it is what gets listened to.
        self.skipped_missing_audio = 0
        if source == "stem":
            present = [s for s in stems if s.stem_path.exists()]
            self.skipped_missing_audio = len(stems) - len(present)
            stems = present

        self.skipped_polyphonic = 0
        kept = []
        for stem in stems:
            if stem.polyphony_rate() > max_polyphony:
                self.skipped_polyphonic += 1
                continue
            kept.append(replace(stem, source=source))
        self._stems = kept

    @property
    def name(self) -> str:
        return f"slakh2100 {self.split} ({self.source})"

    def __iter__(self) -> Iterator[BassStem]:
        return iter(self._stems)

    def __len__(self) -> int:
        return len(self._stems)