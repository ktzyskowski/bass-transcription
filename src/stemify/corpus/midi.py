"""Reading note events out of a MIDI file. One reader, two corpora.

Slakh and FiloBass both ship note-native truth as MIDI, and both had their own copy of
this -- the same deferred import, the same warning suppression, the same non-drum
filter, the same sort. Identical logic in two loaders is the shape that drifts: a fix
to one (a second instrument track, a tempo edge case) silently leaves the other
reporting a reference missing half its notes, which scores a model as hallucinating
rather than as correct.

What stays with each corpus is what is genuinely its own: Slakh's ``BASS_TRANSPOSE``,
FiloBass's lack of one. Those are facts about how each was rendered, not about MIDI.
"""

import warnings
from pathlib import Path

Note = tuple[float, float, int]
"""``(start seconds, end seconds, MIDI pitch)`` -- the raw triple, before transposition.

Deliberately not a ``NoteEvent``. These are pre-transposition and pre-gating: Slakh's
pitches are an octave above where they sound, and both corpora derive their voicing from
a decay gate rather than from note-off. Handing back the corpus's own numbers keeps the
correction visible at the point that knows about it.
"""


def read_notes(midi_path: str | Path) -> list[Note]:
    """Every non-drum note in the file, sorted by onset.

    **Every instrument, not the first.** Each file here holds exactly one non-drum
    instrument, but assuming that would drop an unexpected second track silently --
    and a reference missing half its notes is worse than an error, because it scores
    a correct model as hallucinating.

    Warnings are suppressed because Lakh MIDI is full of harmless oddities (stray key
    signatures, out-of-range tempo events) and they are not this project's to fix; the
    parse succeeds regardless.
    """
    import pretty_midi  # deferred: pulls in mido at import time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    return sorted(
        (n.start, n.end, n.pitch)
        for inst in midi.instruments
        if not inst.is_drum
        for n in inst.notes
    )
