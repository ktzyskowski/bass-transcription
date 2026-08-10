"""``Corpus``: the shape every loader turned out to have -- index, reject, adopt, limit.

Four loaders had four copies of the same loop and disagreed about it in ways nobody had
noticed, because each copy was short enough to look obviously right:

  - **``limit`` meant different things.** Slakh applied it *before* dropping tracks with
    missing audio and tracks that are too polyphonic, so ``--limit 40`` yielded 37 and
    the shortfall depended on the corpus. The other three applied it last and yielded
    exactly N. A run's own track count therefore did not follow from its command line.
  - **Only Slakh counted what it dropped**, and nothing ever read the counters.
  - **The root-existence check** sat inside ``index()`` for three corpora and inside
    ``__init__`` for the fourth, so ``idmt.index(nonexistent)`` returned ``[]`` where the
    others raised.

The differences between corpora are real, but they are *data* -- a predicate and a
mapping -- not structure. Stating them as data is what let the loop be written once.

**``Dataset`` stays a Protocol; this is a base class.** They are different things and
both are wanted: the protocol is the contract a consumer names (``Dataset[NoteAnnotated]``),
and satisfying it is structural, so a two-line test stand-in is still a first-class
dataset. This is shared *implementation*, which is what a base class is actually for. A
loader is free not to use it.

**Laziness is load-bearing, not an optimisation.** ``_reject`` may be expensive --
Slakh's polyphony test parses a MIDI file per track -- and the loop stops as soon as
``limit`` items are kept. That is what lets the limit be applied *after* filtering
without ``--limit 40`` paying for a MIDI parse of all ~1500 tracks in the split. Get
this wrong and the correct semantics cost minutes per construction.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Generic, NoReturn, TypeVar

from stemify.corpus.base import AudioItem

ItemT = TypeVar("ItemT", bound=AudioItem)


def refuse_layout(root: Path, missing: str, expected: str) -> NoReturn:
    """The root exists but does not hold what this corpus expects.

    Separate from "the root does not exist" because the causes are different and so are
    the fixes: a wrong ``--root`` is a typo, whereas this is almost always one extra
    directory level from unpacking the archive. Listing what *is* there is what makes
    that obvious -- the reader sees the nesting immediately instead of re-reading the
    layout in a docstring.
    """
    present = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    raise FileNotFoundError(
        f"Nothing in {root / missing}. Contents of {root}: {present or 'empty'}. "
        f"Expected the archive's own layout: {expected}."
    )


class Corpus(ABC, Generic[ItemT]):
    """A filtered, mapped, limited view of whatever ``_index`` finds on disk."""

    def __init__(self, root: str | Path, limit: int | None = None) -> None:
        """Build the item list.

        **Call this last**, after the subclass has set whatever ``_index``, ``_reject``
        and ``_adopt`` read -- it runs the pipeline immediately, so a field assigned
        after this line is not visible to the filters that needed it. That ordering is
        the one trap in this design; ``tests/corpus/test_loader.py`` constructs every
        real dataset to catch it.
        """
        self.root = Path(root)
        if not self.root.exists():
            # Checked here so all four corpora fail the same way on a typo'd --root.
            # It used to live inside three of the four `index()` functions and in the
            # fourth's `__init__`, so IDMT returned an empty dataset where the others
            # raised -- a wrong path producing an empty run instead of an error.
            raise FileNotFoundError(
                f"{self.root} does not exist -- see docs/datasets.md for the fetch."
            )
        self.skipped: dict[str, int] = {}
        """How many candidates each rejection reason accounted for.

        Keyed by the reason ``_reject`` returned, so the names come from the corpus
        rather than from a fixed vocabulary here. Only populated for candidates actually
        *examined*: with a limit set the loop stops early, so this describes the prefix
        that was read, not the whole corpus. Compare against ``examined``.
        """
        self.examined = 0
        """Candidates considered before the limit was reached.

        The number that makes the limit honest. ``len(dataset)`` says how many tracks a
        run trains on; this says how many it had to look at to find them, and the gap is
        exactly ``sum(skipped.values())``.
        """
        kept: list[ItemT] = []
        for candidate in self._index():
            if limit is not None and len(kept) >= limit:
                break
            self.examined += 1
            reason = self._reject(candidate)
            if reason is not None:
                self.skipped[reason] = self.skipped.get(reason, 0) + 1
                continue
            kept.append(self._adopt(candidate))
        self._items = kept

    # -- what a subclass supplies ------------------------------------------------

    @abstractmethod
    def _index(self) -> Iterable[ItemT]:
        """Every candidate on disk, in a stable order.

        The order is part of the contract: ``limit`` takes a *prefix*, so two runs with
        the same limit must see the same tracks or nothing is comparable.
        """

    def _reject(self, item: ItemT) -> str | None:
        """Why this candidate is not usable, or ``None`` to keep it.

        A short reason rather than a bool, because it becomes the key in ``skipped`` --
        so "40 of 44 kept (3 polyphonic, 1 missing audio)" falls out with no per-corpus
        counter to maintain. Default: keep everything.
        """
        return None

    def _adopt(self, item: ItemT) -> ItemT:
        """The kept candidate as this dataset's own.

        Where a dataset binds its configuration into each item -- Slakh and MDB fix
        ``source`` here so a dataset built for mixes cannot be scored against stems
        partway through a run. Default: pass through.
        """
        return item

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identity, including any variant -- the source or split matters
        as much as the dataset name when reading a result months later."""

    # -- the Dataset protocol ----------------------------------------------------

    def __iter__(self) -> Iterator[ItemT]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
