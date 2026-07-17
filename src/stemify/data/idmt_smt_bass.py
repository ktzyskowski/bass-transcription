"""Loader for the IDMT-SMT-Bass dataset of isolated bass notes.

Layout on disk (see IDMT-SMT-Bass-Description.pdf)::

    idmt_smt_bass/
        PS/{FS,MU,PK,SP,ST}/     plucking styles, expression is always NO
        ES/{BE,DN,HA,NO,VI}/     expression styles, plucking is always FS
        ES/{BEQ,BES,SLD,SLU,     expression subclasses, flattened into ES/ on
            VIF,VIS}/            disk rather than the documented ES_Subclass/

Two filename schemas appear, both encoding the same fields::

    BS_{bass}_EQ_{eq}_{plucking}_{expression}_{string}_{fret}.wav
    BS_{bass}_EQ_{eq}_PS_{plucking}_ES_{expression}_{string}_{fret}.wav

The long form is used by the six subclass folders, the short form everywhere
else.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Bass guitars, keyed by the {bass} field. BS_4 only appears in SLD/SLU.
BASSES = {
    1: "Fame Baphomet NTB 4-string",
    2: "Warwick 4-string",
    3: "Yamaha BB604",
    4: "Jazz Bass 4-string",
}

# Pickup setting, keyed by the {eq} field.
PICKUPS = {1: "neck", 2: "neck+bridge", 3: "bridge"}

PLUCKING_STYLES = {
    "FS": "finger-style",
    "MU": "muted",
    "PK": "picked",
    "SP": "slap-pluck",
    "ST": "slap-thumb",
}

EXPRESSION_STYLES = {
    "NO": "normal",
    "BE": "bending",
    "DN": "dead-note",
    "HA": "harmonics",
    "VI": "vibrato",
    "BEQ": "quarter-tone bending",
    "BES": "semi-tone bending",
    "SLD": "slide down",
    "SLU": "slide up",
    "VIF": "fast vibrato",
    "VIS": "slow vibrato",
}

# Subclasses roll up to a parent expression style. SLD/SLU have no parent class
# of their own -- they are only ever labelled at subclass granularity.
EXPRESSION_PARENT = {
    "BEQ": "BE",
    "BES": "BE",
    "VIF": "VI",
    "VIS": "VI",
    "SLD": None,
    "SLU": None,
}

# The dataset numbers strings 1=E .. 4=G, i.e. low-to-high. This is the reverse
# of the usual tab convention (1=G .. 4=E), so anything downstream that speaks
# tab must convert rather than pass these through.
OPEN_STRING_MIDI = {1: 28, 2: 33, 3: 38, 4: 43}  # E1, A1, D2, G2

_SHORT = re.compile(
    r"^BS_(?P<bass>\d+)_EQ_(?P<eq>\d+)"
    r"_(?P<plucking>[A-Z]+)_(?P<expression>[A-Z]+)"
    r"_(?P<string>\d+)_(?P<fret>\d+)\.wav$"
)
_LONG = re.compile(
    r"^BS_(?P<bass>\d+)_EQ_(?P<eq>\d+)"
    r"_PS_(?P<plucking>[A-Z]+)_ES_(?P<expression>[A-Z]+)"
    r"_(?P<string>\d+)_(?P<fret>\d+)\.wav$"
)


@dataclass(frozen=True)
class Note:
    """A single recorded note and its ground-truth labels."""

    path: Path
    bass: int
    pickup: int
    plucking: str
    expression: str
    string: int
    fret: int

    @property
    def expression_parent(self) -> str | None:
        """Expression style at parent granularity, e.g. BEQ -> BE."""
        return EXPRESSION_PARENT.get(self.expression, self.expression)

    @property
    def midi_pitch(self) -> int | None:
        """Nominal MIDI pitch, or None where fretting does not determine it.

        Harmonics sound above the fretted note, dead notes have no pitch, and
        bends/slides move away from where they start, so a (string, fret) pair
        only pins down the pitch for NO/VI/VIF/VIS.
        """
        if self.expression not in ("NO", "VI", "VIF", "VIS"):
            return None
        open_midi = OPEN_STRING_MIDI.get(self.string)
        return None if open_midi is None else open_midi + self.fret


def parse_filename(path: Path) -> Note | None:
    """Parse a dataset filename, returning None if it does not match a schema."""
    match = _LONG.match(path.name) or _SHORT.match(path.name)
    if match is None:
        return None
    return Note(
        path=path,
        bass=int(match["bass"]),
        pickup=int(match["eq"]),
        plucking=match["plucking"],
        expression=match["expression"],
        string=int(match["string"]),
        fret=int(match["fret"]),
    )


def index(root: str | Path, drop_duplicates: bool = True) -> list[Note]:
    """Index every note in the dataset.

    ES/NO and PS/FS hold 469 byte-identical files -- the same finger-style
    normal notes reached through either taxonomy. ``drop_duplicates`` keeps only
    the PS/FS copy, which matters because otherwise the same recording can land
    on both sides of a train/test split.

    Three files claim string 5 on instruments the documentation lists as
    4-string; they are dropped as mislabelled.
    """
    root = Path(root)
    notes: list[Note] = []
    for path in sorted(root.rglob("*.wav")):
        note = parse_filename(path)
        if note is None:
            continue
        if note.string not in OPEN_STRING_MIDI:
            continue
        if drop_duplicates and path.parent.name == "NO" and note.plucking == "FS":
            continue
        notes.append(note)
    return notes
