"""SlakhDataset: the filtered collection of bass stems a run trains or scores on.

Split out of slakh.py. The *filtering policy* -- which split, which source, the
monophonic premise, the limit -- is what changes here, independently of how the tree is
walked or what one stem holds.
"""

from dataclasses import replace
from pathlib import Path

from stemify.corpus.loader import Corpus
from stemify.corpus.slakh.index import index
from stemify.corpus.slakh.item import Item as _Item


class SlakhDataset(Corpus[_Item]):
    """Slakh2100 bass stems, filtered to those the monophonic premise holds for.

    Defaults to the ``test`` split. Slakh is large enough to train on, so
    evaluating on ``train`` reports a number that stops meaning anything the
    moment a model exists.

    Stems whose MIDI is substantially polyphonic are dropped: a doubled-octave
    transcription has no single right answer, and scoring against one would
    manufacture octave errors that are the transcription's fault rather than the
    model's -- especially misleading in a project whose central metric is the
    octave-error rate.

    The polyphony filter reads every candidate's MIDI, so construction is the
    expensive part and iteration is cheap. Pass ``limit`` to cut that cost while
    iterating.
    """

    Item = _Item
    """The item type this dataset yields, reachable from the dataset that yields it.

    An alias, not a nested class: ``Item`` is defined at module scope so it stays
    picklable by a plain qualified name and, for Slakh, so ``index`` can build one
    without importing the dataset that indexes it. Both spellings work -- ``slakh.Item``
    when you have the module, ``SlakhDataset.Item`` when you have the dataset.
    """


    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        source: str = "stem",
        limit: int | None = None,
        max_polyphony: float = 0.05,
        unambiguous_only: bool | None = None,
    ) -> None:
        if source not in ("stem", "mix"):
            raise ValueError(f"source must be 'stem' or 'mix', got {source!r}")
        self.split = split
        self.source = source
        self.max_polyphony = max_polyphony
        # Mixes *require* single-bass tracks -- with two basses sounding, one
        # reference contour cannot describe the mix. Stems do not, so the two
        # sources otherwise yield different track sets (147 vs 124 on test),
        # which would silently make a stem-vs-mix comparison uncontrolled.
        # Pass unambiguous_only=True explicitly to hold the set fixed across
        # conditions.
        if unambiguous_only is None:
            unambiguous_only = source == "mix"
        self.unambiguous_only = unambiguous_only
        super().__init__(root, limit)  # last: the filters below read the fields above

    def _index(self) -> list[_Item]:
        return index(
            self.root, splits=(self.split,), unambiguous_only=self.unambiguous_only
        )

    def _reject(self, item: _Item) -> str | None:
        # Stem audio is only required when it is what gets listened to.
        if self.source == "stem" and not item.stem_path.exists():
            return "missing audio"
        # Expensive -- a MIDI parse per track -- which is why the base stops as soon as
        # `limit` items are kept rather than filtering the whole split first.
        if item.polyphony_rate() > self.max_polyphony:
            return "polyphonic"
        return None

    def _adopt(self, item: _Item) -> _Item:
        return replace(item, source=self.source)

    @property
    def name(self) -> str:
        return f"slakh2100 {self.split} ({self.source})"
