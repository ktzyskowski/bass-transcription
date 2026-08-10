"""Which corpora satisfy which contract -- the classification the whole design rests on.

``AudioItem`` and ``NoteAnnotated`` exist to make one distinction checkable: every corpus
here has a frame-level pitch contour, but only Slakh and FiloBass have note-native truth.
Everything that trains or scores at the note level asks for ``Dataset[NoteAnnotated]``,
so if this classification is wrong the type system is confidently wrong -- which is worse
than the ``NotImplementedError`` it replaced.

Items are constructed directly with paths that need not exist: these are frozen
dataclasses that touch the filesystem only when a reference is asked for, which is the
laziness the contract promises. That is what lets this run with no corpus on disk and no
fakes -- the pillar test.
"""

from pathlib import Path

import pytest

from stemify.corpus.base import AudioItem, NoteAnnotated
from stemify.corpus.filobass import FiloBassDataset
from stemify.corpus.idmt_smt_bass import IdmtDataset
from stemify.corpus.mdb_bass_synth import MdbDataset
from stemify.corpus.slakh import SlakhDataset

NOTE_NATIVE = {
    "slakh": SlakhDataset.Item(
        track="Track00042", split="train", stem_id="S03", stem_path=Path("a.flac"),
        midi_path=Path("a.mid"), mix_path=Path("mix.flac"),
        program_name="Electric Bass (finger)", n_bass_stems=1,
    ),
    "filobass": FiloBassDataset.Item(
        tune="Autumn Leaves", audio_path=Path("a.mp3"), midi_path=Path("a.mid")
    ),
}

FRAME_ONLY = {
    "mdb": MdbDataset.Item(
        track_name="song", stem_path=Path("a.wav"), annotation_path=Path("a.csv")
    ),
    "idmt": IdmtDataset.Item(
        path=Path("BS_1_EQ_1_FS_NO_1_00.wav"), bass=1, pickup=1, plucking="FS",
        expression="NO", string=1, fret=0,
    ),
}

EVERY_CORPUS = {**NOTE_NATIVE, **FRAME_ONLY}


@pytest.mark.parametrize("item", EVERY_CORPUS.values(), ids=EVERY_CORPUS)
def test_every_corpus_item_is_an_eval_item(item):
    """The universal contract: identity, audio roles, strata, and a contour.

    Structural -- none of these inherit from anything. That is the point: the ABC this
    replaced was an inheritance edge that only the real loaders honoured, while every
    test fixture duck-typed past it.
    """
    assert isinstance(item, AudioItem)


@pytest.mark.parametrize("item", NOTE_NATIVE.values(), ids=NOTE_NATIVE)
def test_note_native_corpora_carry_note_truth(item):
    """Slakh is rendered from MIDI, FiloBass is transcribed to it. Both can answer
    ``reference_notes()``, so both are trainable and scoreable at the note level."""
    assert isinstance(item, NoteAnnotated)


@pytest.mark.parametrize("item", FRAME_ONLY.values(), ids=FRAME_ONLY)
def test_frame_annotated_corpora_do_not_claim_note_truth(item):
    """The half that used to be a lie.

    MDB-bass-synth is annotated with a resynthesis control signal and IDMT with a
    ``(string, fret)`` label -- neither has note boundaries. They previously *did*
    satisfy the note interface, because the base class gave everyone a
    ``reference_notes()`` that raised, so "has note truth" and "will raise if you ask"
    were the same shape and only the second was discoverable, at runtime, mid-run.
    """
    assert not isinstance(item, NoteAnnotated)
    assert not hasattr(item, "reference_notes")


@pytest.mark.parametrize("item", EVERY_CORPUS.values(), ids=EVERY_CORPUS)
def test_naming_an_item_touches_no_files(item):
    """References are lazy and identity is not.

    ``name``, ``groups`` and ``sources()`` are answerable from the index alone, on paths
    that do not exist here. A run that filters most items out must not pay a MIDI parse
    per item to find that out -- which is why ``reference()`` is a method rather than a
    field, and why indexing a corpus is cheap.
    """
    assert isinstance(item.name, str) and item.name
    assert isinstance(item.groups, dict)
    assert item.sources().bass
