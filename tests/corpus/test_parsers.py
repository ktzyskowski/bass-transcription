"""The parsers: filename schemas, file pairing, and what happens on a wrong root.

The part of a corpus loader that actually goes wrong. A layout misread does not raise --
it indexes zero tracks, or pairs an annotation with the wrong audio, and the run that
follows reports a number for an experiment nobody ran. All of it is testable against
synthetic trees, so none of these need a corpus on disk.
"""

from pathlib import Path

import pytest

from stemify.corpus import idmt_smt_bass as idmt
from stemify.corpus import mdb_bass_synth as mdb
from stemify.corpus.idmt_smt_bass import IdmtDataset
from stemify.corpus.mdb_bass_synth import MdbDataset


# --------------------------------------------------------------------------- #
# IDMT: two filename schemas encoding the same fields                        #
# --------------------------------------------------------------------------- #
def test_the_short_schema_parses():
    got = idmt.parse_filename(Path("BS_2_EQ_1_FS_NO_3_07.wav"))
    assert got is not None
    assert (got.bass, got.pickup, got.plucking) == (2, 1, "FS")
    assert (got.expression, got.string, got.fret) == ("NO", 3, 7)


def test_the_long_schema_parses_to_the_same_fields():
    """The six subclass folders use ``PS_``/``ES_`` prefixes; everything else does not.
    Both encode identical fields, so a reader that handled only one would silently index
    a fraction of the corpus."""
    short = idmt.parse_filename(Path("BS_2_EQ_1_FS_NO_3_07.wav"))
    long = idmt.parse_filename(Path("BS_2_EQ_1_PS_FS_ES_NO_3_07.wav"))
    assert short == long.__class__(**{**long.__dict__, "path": short.path})


def test_a_name_matching_neither_schema_is_declined_not_guessed():
    assert idmt.parse_filename(Path("README.txt")) is None
    assert idmt.parse_filename(Path("BS_2_EQ_1_FS_NO_3.wav")) is None  # no fret field


def test_fret_zero_is_the_open_string_not_a_missing_field():
    got = idmt.parse_filename(Path("BS_1_EQ_1_FS_NO_1_00.wav"))
    assert got is not None and got.fret == 0
    assert got.midi_pitch == idmt.OPEN_STRING_MIDI[1]


@pytest.mark.parametrize("expression", ["HA", "DN", "SLD", "SLU", "BE"])
def test_expressions_that_do_not_determine_a_pitch_report_none(expression):
    """Harmonics sound above the fretted note, dead notes have no pitch, bends and
    slides move away from where they start. ``(string, fret)`` only pins the pitch for
    NO/VI/VIF/VIS -- and `IdmtDataset` drops the rest rather than scoring a baseline on
    a question the corpus cannot answer."""
    got = idmt.parse_filename(Path(f"BS_1_EQ_1_FS_{expression}_1_05.wav"))
    assert got is not None and got.midi_pitch is None


def test_a_root_that_does_not_exist_raises():
    """It used to return ``[]`` for IDMT while the other three raised, so a typo'd
    ``--root`` produced an empty run rather than an error. The check now lives once, in
    ``Corpus``, so all four fail identically."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        IdmtDataset(Path("/nonexistent/idmt"))


# --------------------------------------------------------------------------- #
# MDB: three files per track that do NOT share a prefix                      #
# --------------------------------------------------------------------------- #
def _mdb_tree(root: Path, songs, *, with_mix=True):
    for d in (mdb.ANNOTATION_DIR, mdb.BASS_DIR, mdb.MIX_DIR):
        (root / d).mkdir(parents=True, exist_ok=True)
    for song, stem_id in songs:
        base = f"{song}_STEM_{stem_id}.RESYN"
        (root / mdb.ANNOTATION_DIR / f"{base}.csv").write_text("0.0,110.0\n")
        (root / mdb.BASS_DIR / f"{base}.wav").write_bytes(b"")
        if with_mix:
            (root / mdb.MIX_DIR / f"{song}_MIX_basssynth.wav").write_bytes(b"")
    return root


def test_the_three_files_are_paired_by_stripping_the_stem_id(tmp_path):
    """The mix filename drops the ``_STEM_<id>.RESYN`` the other two carry, so they
    cannot be found by a shared glob. Getting this wrong pairs an annotation with
    another song's mix -- labels intact, audio wrong, loss still falling."""
    _mdb_tree(tmp_path, [("ArtistA_SongOne", "03")])
    (item,) = MdbDataset(tmp_path)
    assert item.track_name == "ArtistA_SongOne"
    assert item.annotation_path.name == "ArtistA_SongOne_STEM_03.RESYN.csv"
    assert item.stem_path.name == "ArtistA_SongOne_STEM_03.RESYN.wav"
    assert item.mix_path is not None
    assert item.mix_path.name == "ArtistA_SongOne_MIX_basssynth.wav"


def test_a_missing_mix_is_absent_rather_than_a_broken_path(tmp_path):
    """Not every track ships a mix. ``mix_path`` is ``None`` so a bichannel condition
    fails where the channel is assembled, rather than handing a reader a path to a file
    that is not there."""
    _mdb_tree(tmp_path, [("ArtistA_SongOne", "03")], with_mix=False)
    (item,) = MdbDataset(tmp_path)
    assert item.mix_path is None


def test_an_annotation_with_no_audio_is_dropped(tmp_path):
    _mdb_tree(tmp_path, [("A_One", "01")])
    (tmp_path / mdb.BASS_DIR / "A_One_STEM_01.RESYN.wav").unlink()
    assert len(MdbDataset(tmp_path)) == 0


def test_a_root_with_the_wrong_layout_says_what_it_found(tmp_path):
    """The error a mis-unpacked archive produces -- distinct from "root does not exist",
    because the cause is different: almost always one extra directory level. It names
    what *is* there, so the nesting is visible instead of needing the layout re-read."""
    (tmp_path / "MDB-bass-synth").mkdir()
    with pytest.raises(FileNotFoundError, match="Expected the archive's own layout"):
        MdbDataset(tmp_path)


def test_the_annotation_reader_tolerates_a_header_and_comments(tmp_path):
    """Copies in the wild have both. A parser that fell over on a header would be
    caught immediately; one that parsed the header *as data* yields a 0 Hz frame at
    time 0, which reads as a plausible unvoiced first frame."""
    _mdb_tree(tmp_path, [("A_One", "01")])
    (tmp_path / mdb.ANNOTATION_DIR / "A_One_STEM_01.RESYN.csv").write_text(
        "# generated\ntime,frequency\n0.00,0.0\n0.01,110.0\n"
    )
    (item,) = MdbDataset(tmp_path)
    contour = item.reference_contour()
    assert len(contour) == 2
    assert list(contour.freqs) == [0.0, 110.0]
