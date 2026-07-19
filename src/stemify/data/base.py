"""What every dataset must look like to be evaluable.

Previously each loader exposed its own type and its own module-level ``index()``,
and the harness bridged them with one adapter function per dataset plus a branch
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


class EvalItem(ABC):
    """One scorable unit: some audio and the pitch truth to score it against."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier, unique within its dataset. Shown in per-item breakdowns."""

    @abstractmethod
    def audio(self) -> Path:
        """The audio a baseline should listen to."""

    @abstractmethod
    def reference(self) -> Contour:
        """Ground-truth pitch over time. Built on demand, not at index time."""

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
