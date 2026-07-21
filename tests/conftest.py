"""Shared fixtures for the test suite.

The metrics operate on ``NoteEvent`` lists, so almost every test builds a few by
hand. ``note`` is a terse factory for that -- ``note(onset, offset, midi)`` --
so the interesting part of a case (the onsets and pitches that make it match or
not) stays legible instead of being buried in ``NoteEvent(...)`` boilerplate.
"""

from collections.abc import Callable

import pytest

from stemify.notes import NoteEvent


@pytest.fixture
def note() -> Callable[[float, float, float], NoteEvent]:
    def make(onset: float, offset: float, midi: float) -> NoteEvent:
        return NoteEvent(onset=onset, offset=offset, midi_pitch=midi)

    return make