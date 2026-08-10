"""Loader for FiloBass: 48 jazz standards played by professional upright bassists.

The first **out-of-domain** set in this pipeline, and that is its whole point. Every
other corpus here is either isolated notes (IDMT), resynthesized bass (MDB), or bass
rendered from MIDI by a sample library (Slakh). FiloBass is 12 named human bassists
playing real double bass on real recordings, ~4.9 h and 52,558 notes. A model trained
on Slakh has never heard this instrument.

**These stems are `separated`, not `clean`, and the distinction decides which number
they are compared against.** The bassists are the Aebersold session players; nobody
recorded them for this dataset. The Aebersold releases carry bass+drums on one stereo
channel, and the paper takes that channel through **Demucs** to obtain the stem shipped
here -- the same separator this project's `separated` condition uses
(``demucs-htdemucs-bass``). So the right baseline is a model's `separated` score, not
its `clean` one; against `clean` the transfer gap reads ~10 points worse than it is.
Audibly confirmed too: the stems carry essentially nothing above 4 kHz, which is
separator output rather than a mic'd upright.

It is also the densest task available: walking basslines run ~3.1 notes/s with a
~400 ms median duration, against Slakh's sparser lines. The expectation was that this
would stress *segmentation*; measured, it stresses **pitch** -- 64-72% of false
positives are notes in roughly the right place with the wrong pitch, against 13-15%
fragmentation, and that holds across three independently trained checkpoints.

Layout, verified against an unpacked copy::

    filobass/
        audio_bass_stems/<Tune-Name>.mp3        44.1 kHz stereo, ~6 min
        midi_fully_aligned/<Tune-Name>.mid      note-level, aligned to the audio
        midi_downbeat_aligned/<Tune-Name>.mid   aligned at downbeats only
        midi_from_score/<Tune-Name>.mid         score timing, not audio timing
        syncpoints/<Tune-Name>.json             [[bar_index, seconds], ...]
        notebooks_and_scripts/note_data.csv     per-note chord/degree/measure + player

One file per tune in every directory, sharing an exact stem -- no id-stripping, unlike
MDB. The archive nests everything under a directory whose name contains a space
(``FiloBass ISMIR Publication``); flatten it away rather than quoting the path
forever. See docs/datasets.md.

**``midi_fully_aligned`` is the only usable MIDI.** The other two are notation
timing: ``midi_from_score`` carries none of the performance, and
``midi_downbeat_aligned`` is anchored only at bar lines, so its onsets drift by most
of a beat inside each bar. Against a +-50 ms note tolerance both are noise. Checked:
the fully-aligned MIDI's span tracks the audio to within ~2.5 s over ~5.5 min tracks.

**No transposition.** FiloBass MIDI is annotated at *sounding* pitch -- measured range
E1..G3 (MIDI 28-55), median 38-40, occasionally to C4 -- which is where a double bass
actually sounds. Slakh needs ``BASS_TRANSPOSE = 12`` because its MIDI is written an
octave high; applying that here would put every label an octave low, and the error
would be as consistent (and as invisible) as the one that constant exists to fix.

**No mixture, and that is a licensing accident rather than a property of the corpus.**
The archive ships stems only, so ``sources()`` offers no ``mix`` role and a bichannel
checkpoint cannot be evaluated here without silencing its context channel. The missing
audio is ``Bass_Drums.wav`` -- **the exact mixture these stems were separated from** --
which FiloSax's ``Compile_Backing.py`` rebuilds from commercially purchased Aebersold
play-alongs (~25-35 volumes, MD5-verified, so only the exact files work). Obtaining it
would make this a true ``bichannel-separated`` set: separated bass in channel 0, its own
source mixture in channel 1, which is the deployment condition rather than an analogue
of it.

``Piano_Drums.wav`` is the wrong channel for that and the mistake is worth naming: it
holds no bass at all, while ``Bass_Drums`` holds *this* performance -- not a second
bassist, as it would if the stems had been separately recorded.
"""

import csv
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from stemify.music.contour import Contour
from stemify.corpus.base import Dataset, EvalItem
from stemify.corpus.midi_contour import (
    NOTE_DECAY_DB, REFERENCE_HOP_S, notes_to_contour,
)
from stemify.music.notes import NoteEvent

AUDIO_DIR = "audio_bass_stems"
MIDI_DIR = "midi_fully_aligned"
METADATA_CSV = "notebooks_and_scripts/note_data.csv"


@dataclass(frozen=True)
class Track(EvalItem):
    """One tune: a Demucs-separated upright bass stem and its audio-aligned MIDI."""

    tune: str
    audio_path: Path
    midi_path: Path
    bassist: str = ""

    @property
    def name(self) -> str:
        return self.tune

    @property
    def groups(self) -> dict[str, str]:
        """Sliced by player, which is the stratification that means something here.

        The 48 tunes are shared across 12 bassists with quite different tone and
        articulation, so a per-player breakdown is the closest thing this corpus has
        to an ablation over playing style -- and it costs nothing, since the report
        machinery slices by whatever keys it finds.
        """
        return {"bassist": self.bassist} if self.bassist else {}

    def audio(self) -> Path:
        return self.audio_path

    @cached_property
    def _notes(self) -> list[tuple[float, float, int]]:
        """``(start, end, pitch)`` for every note, sorted by onset.

        Every file holds exactly one non-drum instrument, but the comprehension does
        not assume it: an unexpected second track would otherwise be dropped
        silently, and a reference missing half its notes scores a model as
        hallucinating rather than as correct.
        """
        import pretty_midi  # deferred: pulls in mido at import time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            midi = pretty_midi.PrettyMIDI(str(self.midi_path))
        return sorted(
            (n.start, n.end, n.pitch)
            for inst in midi.instruments
            if not inst.is_drum
            for n in inst.notes
        )

    def reference(
        self, hop_s: float = REFERENCE_HOP_S, decay_db: float = NOTE_DECAY_DB
    ) -> Contour:
        """Frame-level ``(times, freqs)``, 0 Hz where no bass sounds.

        The same decay gate Slakh uses, and for the same reason -- an upright's
        sustain outlives its note-off at least as much as a rendered electric's. No
        transposition; see the module docstring.
        """
        return notes_to_contour(self._notes, self.audio_path, 0, hop_s, decay_db)

    def reference_notes(self) -> list[NoteEvent]:
        """The MIDI notes as events -- note-native truth, not a segmented contour.

        ~5% of consecutive pairs overlap (a ringing string under the next attack
        rather than a double stop). They are passed through rather than merged:
        the note metric matches estimates against references independently, so an
        overlap costs at most one unreachable note, whereas silently dropping one
        would make a correct detection look like a false positive.
        """
        return [
            NoteEvent(onset=start, offset=end, midi_pitch=float(pitch))
            for start, end, pitch in self._notes
        ]


def _bassists(root: Path) -> dict[str, str]:
    """``tune -> player`` from ``note_data.csv``; empty if it is missing.

    Optional on purpose. The per-note CSV is analysis output rather than part of the
    annotation, so a copy without it should still evaluate -- it simply loses the
    per-player breakdown instead of failing to index at all.
    """
    path = root / METADATA_CSV
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            tune, player = row.get("title"), row.get("player")
            if tune and player:
                out.setdefault(tune, player)
    return out


def index(root: str | Path) -> list[Track]:
    """Index every tune under ``root``, driven by the aligned MIDI.

    MIDI-first rather than audio-first: a stem without a transcription cannot be
    scored, so it is not an item, while a transcription without audio is a broken
    copy worth naming.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. FiloBass is a single zip from "
            "https://zenodo.org/records/10069709 -- see docs/datasets.md"
        )

    midis = sorted((root / MIDI_DIR).glob("*.mid"))
    if not midis:
        present = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
        raise FileNotFoundError(
            f"No MIDI in {root / MIDI_DIR}. Contents of {root}: {present or 'empty'}. "
            f"Expected the archive's own layout ({AUDIO_DIR}/, {MIDI_DIR}/, ...) with "
            "the 'FiloBass ISMIR Publication' directory flattened away."
        )

    players = _bassists(root)
    tracks, orphans = [], []
    for midi in midis:
        audio = root / AUDIO_DIR / f"{midi.stem}.mp3"
        if not audio.exists():
            orphans.append(midi.name)
            continue
        tracks.append(
            Track(
                tune=midi.stem,
                audio_path=audio,
                midi_path=midi,
                bassist=players.get(midi.stem, ""),
            )
        )

    if orphans:
        print(f"warning: {len(orphans)} transcriptions without audio, e.g. {orphans[:3]}")
    return tracks


class FiloBassDataset(Dataset):
    """FiloBass, evaluated on the Demucs-separated bass stems.

    No ``source`` parameter, unlike MDB and Slakh: only one rendering ships. That is a
    licensing artefact rather than a simplification here -- the mixture these stems came
    from exists, it is just behind a paywall. See the module docstring.
    """

    def __init__(self, root: str | Path, limit: int | None = None) -> None:
        tracks = index(root)
        self._tracks = tracks[:limit] if limit else tracks

    @property
    def name(self) -> str:
        # Names the *state* of the audio, the way `separated` does in core/conditions.py.
        # A reader comparing this against a Slakh number has to know which of
        # `clean`/`separated` it lines up with, and "bass stems" read as clean for a
        # whole session of analysis before anyone checked the paper.
        return "filobass (separated)"

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)
