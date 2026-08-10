"""``NoteEvent`` -- the one invariant a note carries, and the one it does not.

Small enough that the interesting question is not what the fields hold but what the type
refuses. It refuses exactly one thing, and the tests below say why that one and not the
others.
"""

import pytest

from stemify.music.notes import NoteEvent


def test_a_note_may_not_end_before_it_starts():
    with pytest.raises(ValueError, match="ends before it starts"):
        NoteEvent(onset=1.0, offset=0.5, midi_pitch=40.0)


def test_a_zero_length_note_is_allowed():
    """The boundary is ``>=``, not ``>``. A MIDI note-on immediately followed by its
    note-off is degenerate but real, and it is the corpus's fact to report -- refusing it
    here would turn a faithfully-read reference into a crash mid-epoch."""
    assert NoteEvent(onset=1.0, offset=1.0, midi_pitch=40.0).offset == 1.0


def test_pitch_and_confidence_are_not_range_checked():
    """Deliberate, and worth pinning so nobody adds it as an oversight.

    ``midi_pitch`` is a float because estimates land between semitones, and the useful
    range is the *corpus's*, not MIDI's -- a bass model has no business rejecting pitch
    141 on principle when the real constraint (does it fall in this ``LabelSet``) is
    checked where the alphabet is known. Same for ``confidence``: only some producers
    populate it meaningfully, so a [0, 1] assertion here would fire on the ones that use
    it as a raw score.
    """
    assert NoteEvent(onset=0.0, offset=1.0, midi_pitch=141.0, confidence=7.5)


def test_notes_are_frozen_and_compare_by_value():
    """Value semantics are load-bearing: references and estimates are put into sets and
    compared elementwise all over the metrics and the gold builders."""
    a = NoteEvent(onset=0.0, offset=1.0, midi_pitch=40.0)
    assert a == NoteEvent(onset=0.0, offset=1.0, midi_pitch=40.0)
    with pytest.raises(AttributeError):
        a.onset = 2.0  # type: ignore[misc]
