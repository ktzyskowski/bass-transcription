"""Shared fixtures for the test suite.

The metrics operate on ``NoteEvent`` lists, so almost every test builds a few by
hand. ``note`` is a terse factory for that -- ``note(onset, offset, midi)`` --
so the interesting part of a case (the onsets and pitches that make it match or
not) stays legible instead of being buried in ``NoteEvent(...)`` boilerplate.

``device`` sweeps a test over both backends; see its docstring for why a CPU-only
suite is structurally blind to a whole class of bug.
"""

from collections.abc import Callable

import pytest
import torch

from stemify.music.notes import NoteEvent


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Run the test once per available backend.

    Device bugs do not show up on CPU: a tensor built without an explicit ``device=``
    silently agrees with every other CPU tensor, so code that strands a field on the
    host passes a CPU-only suite and then fails on the first line of a CUDA run. Only
    actually placing the module elsewhere exposes it.

    Skips rather than fails when there is no accelerator, so the suite still runs
    fully on a laptop -- but on a CUDA box this is the only thing exercising the path
    that training actually takes."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    return torch.device(request.param)


@pytest.fixture
def note() -> Callable[[float, float, float], NoteEvent]:
    def make(onset: float, offset: float, midi: float) -> NoteEvent:
        return NoteEvent(onset=onset, offset=offset, midi_pitch=midi)

    return make
