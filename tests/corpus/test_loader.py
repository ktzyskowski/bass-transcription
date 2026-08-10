"""``Corpus``: index -> reject -> adopt -> limit, and the three things that got fixed.

The loop was copied four times and the copies disagreed. What is pinned here is the
agreement, plus the property that makes the correct ``limit`` semantics affordable:
rejection may be expensive, so the loop must stop rather than filter the whole corpus.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from stemify.audio.sources import Sources
from stemify.corpus.loader import Corpus


@dataclass(frozen=True)
class _Item:
    n: int
    ok: bool = True
    tag: str = ""

    @property
    def name(self) -> str:
        return f"item{self.n}"

    @property
    def groups(self) -> dict[str, str]:
        return {}

    def sources(self) -> Sources:
        return Sources(bass=Path(f"{self.n}.wav"))


class _Corpus(Corpus[_Item]):
    """Every candidate is countable and rejection is *observable* -- the test needs to
    know not just what survived but what was looked at."""

    def __init__(self, root, items, limit=None, tag=""):
        self.candidates = list(items)
        self.touched: list[int] = []
        self.tag = tag
        super().__init__(root, limit)

    def _index(self):
        return self.candidates

    def _reject(self, item):
        self.touched.append(item.n)          # records the cost of examining
        return None if item.ok else "not ok"

    def _adopt(self, item):
        return replace(item, tag=self.tag)

    @property
    def name(self) -> str:
        return "fake"


def _items(*flags):
    return [_Item(n=i, ok=f) for i, f in enumerate(flags)]


def test_a_limit_yields_exactly_that_many_despite_rejections(tmp_path):
    """The fix. Slakh applied the limit *before* filtering, so ``--limit 40`` returned
    37 and the shortfall depended on the corpus -- a run's track count did not follow
    from its command line."""
    corpus = _Corpus(tmp_path, _items(True, False, False, True, True, True), limit=3)
    assert len(corpus) == 3


def test_rejection_stops_as_soon_as_the_limit_is_met(tmp_path):
    """Laziness is load-bearing, not an optimisation. Slakh's ``_reject`` parses a MIDI
    file per track; filtering the whole split before limiting would make ``--limit 40``
    pay for all ~1500 of them. Nothing past the last kept item may be touched."""
    corpus = _Corpus(tmp_path, _items(True, True, True, True, True, True), limit=2)
    assert corpus.touched == [0, 1]
    assert corpus.examined == 2


def test_a_limit_larger_than_the_corpus_is_not_an_error(tmp_path):
    corpus = _Corpus(tmp_path, _items(True, True), limit=99)
    assert len(corpus) == 2


def test_no_limit_examines_everything(tmp_path):
    corpus = _Corpus(tmp_path, _items(True, False, True), limit=None)
    assert corpus.examined == 3 and len(corpus) == 2


def test_skips_are_counted_by_reason(tmp_path):
    """Named by the corpus, not from a fixed vocabulary here -- which is what lets
    "40 of 44 kept (3 polyphonic, 1 missing audio)" fall out with no per-corpus counter
    to maintain. Only Slakh had counters before, and nothing read them."""
    corpus = _Corpus(tmp_path, _items(True, False, False, True))
    assert corpus.skipped == {"not ok": 2}


def test_examined_minus_kept_is_exactly_what_was_skipped(tmp_path):
    """The invariant that makes the limit honest: `len` is what a run trains on,
    `examined` is what it had to look at, and the gap is fully accounted for."""
    corpus = _Corpus(tmp_path, _items(True, False, True, False, True), limit=3)
    assert corpus.examined - len(corpus) == sum(corpus.skipped.values())


def test_nothing_is_skipped_by_default(tmp_path):
    class _KeepAll(_Corpus):
        def _reject(self, item):
            return None

    assert _KeepAll(tmp_path, _items(True, False)).skipped == {}


def test_adopt_binds_the_datasets_configuration_into_each_item(tmp_path):
    """Where Slakh and MDB fix ``source``, so a dataset built for mixes cannot be
    scored against stems partway through a run."""
    corpus = _Corpus(tmp_path, _items(True, True), tag="mix")
    assert [i.tag for i in corpus] == ["mix", "mix"]


def test_a_subclass_must_supply_index_and_name(tmp_path):
    """Abstract rather than a raising default: the failure belongs at construction, not
    at the first iteration halfway through a run."""
    with pytest.raises(TypeError):
        Corpus(tmp_path)  # type: ignore[abstract]


def test_a_root_that_does_not_exist_is_refused_before_any_indexing(tmp_path):
    """Shared, so all four corpora fail the same way on a typo'd --root -- and *before*
    `_index` runs, so a loader never has to answer what it found in a directory that is
    not there."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _Corpus(tmp_path / "nope", _items(True))
