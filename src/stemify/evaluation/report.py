"""The shared surface of an evaluation result.

A run's results are data, not printed output: the previous harness printed and
returned nothing, so a result existed only in a scrollback -- impossible to diff
across runs or read back from a stored job. Every report is therefore
serialisable and renderable.

Frame-level and note-level evaluations produce different reports (``FrameReport``
in ``frame``, ``NoteReport`` in ``note``) -- they carry different numbers and lay
themselves out differently. What they share is this surface: ``render`` for a
human, ``to_dict`` for a machine, and ``write_json`` on top of ``to_dict``. The
base supplies ``write_json`` once so neither subclass repeats it, and lets a
caller (the CLI, a comparison script) hold either report behind one type.

The base is intentionally thin. Aggregation and formatting are level-specific and
live with each subclass; forcing them into a shared template would abstract over
almost nothing, since a frame report pools metric results by group while a note
report pools tp/fp/fn counts.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays to something ``json`` accepts. Shared by both
    report types, which both serialise numbers that may arrive as numpy dtypes."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class Report(ABC):
    """What one run scored, as a diffable, renderable object.

    Subclasses implement ``render`` (human-readable) and ``to_dict`` (a plain
    JSON-able dict); ``write_json`` comes for free from ``to_dict``.
    """

    @abstractmethod
    def render(self, worst: int = 10) -> str:
        """A human-readable summary. ``worst`` bounds the worst-items tail."""

    @abstractmethod
    def to_dict(self) -> dict:
        """A plain dict suitable for ``json.dumps`` -- the machine-readable form."""

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))