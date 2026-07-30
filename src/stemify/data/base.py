"""What every dataset must look like to be evaluable.

Previously each loader exposed its own type and its own module-level ``index()``,
and the runner bridged them with one adapter function per dataset plus a branch
in ``main``. Adding a dataset meant editing the runner in three places -- and for
IDMT, the runner also had to know *how to build that dataset's reference*, which
is dataset knowledge sitting in the wrong file.

The interface below was not invented for this module. MDB's ``Track`` and Slakh's
``BassStem`` had independently converged on ``.name``, ``.audio()`` and
``.reference()`` before it existed; this only declares what they already were.

Two design points worth stating:

**References are lazy.** ``reference()`` is a method, not a field, because
building one can be expensive -- Slakh parses a MIDI file per item -- and a run
that stops early, or filters most items out, should not pay for all of them up
front.

**Grouping is the item's business.** An item declares its own ``groups``
("string 1", "plucking FS", "split test"), and the report machinery slices by
whatever keys it finds. That is how per-string and per-technique breakdowns
survive without the evaluation layer knowing anything about bass guitars.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from stemify.contour import Contour
from stemify.notes import NoteEvent
from stemify.sources import Sources


class EvalItem(ABC):
    """One scorable unit: some audio and the pitch truth to score it against."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier, unique within its dataset. Shown in per-item breakdowns."""

    @abstractmethod
    def audio(self) -> Path:
        """The audio a baseline should listen to."""

    def sources(self) -> Sources:
        """Every audio role this item can offer, not just the one being listened to.

        ``audio()`` answers "which single file is this item *about*"; this answers
        "what else is on hand for the same music". They differ only for a dataset
        that ships more than one rendering -- Slakh has the mixture beside every
        stem -- and that difference is exactly what a multi-channel encoder reads:
        the bass in channel 0, the mixture it sits inside in channel 1.

        The default keeps the two the same, so a single-file dataset needs no
        change and simply has no second channel to offer.
        """
        return Sources(bass=self.audio())

    @abstractmethod
    def reference(self) -> Contour:
        """Ground-truth pitch over time. Built on demand, not at index time."""

    def reference_notes(self) -> list[NoteEvent]:
        """Ground-truth note events, for note-level evaluation.

        Optional: only datasets with note-native truth (Slakh's MIDI, IDMT's
        one-note clips) implement this. A frame-annotated dataset like
        MDB-bass-synth has no note boundaries to report and stays frame-level.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no note-level reference"
        )

    @property
    def groups(self) -> dict[str, str]:
        """Labels this item can be broken down by, e.g. ``{"string": "1"}``.

        Empty by default: a dataset with no meaningful strata simply reports its
        aggregate.
        """
        return {}


class Dataset(ABC):
    """An iterable of evaluable items, owning how its references are built."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identity, including any variant -- the source or split
        matters as much as the dataset name when reading a result months later."""

    @abstractmethod
    def __iter__(self) -> Iterator[EvalItem]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
