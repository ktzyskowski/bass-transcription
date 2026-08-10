"""What a corpus must look like to be usable -- split by what it can actually answer.

Previously each loader exposed its own type and its own module-level ``index()``, and the
runner bridged them with one adapter function per dataset plus a branch in ``main``.
Adding a dataset meant editing the runner in three places. The interface below was not
invented to fix that: MDB's ``Track`` and Slakh's ``BassStem`` had independently
converged on ``.name``, ``.sources()`` and ``.reference_contour()`` before it existed. It only
declares what they already were.

**Protocols, not base classes.** Nothing here needs to be inherited from. Every loader
satisfies these structurally, every test fixture already duck-types them, and the
``ABC`` this replaced was buying an inheritance edge and no enforcement -- the fakes
never subclassed it, so the abstractness only ever constrained the real implementations.
Structural typing checks the same thing at every point that matters (the call site) and
lets a two-line stand-in in a test be a first-class item.

**Two protocols, because there are two kinds of truth.** This is the correction that
motivated the split. Every corpus here has a frame-level pitch contour; only Slakh and
FiloBass have note-native truth. That was previously expressed by making
``reference()`` abstract and ``reference_notes()`` a method that *raises* -- so the
optional member was the one the semi-CRF, the trainer's validator and both evaluation
levels actually call, and asking for it from MDB or IDMT was a ``NotImplementedError``
raised deep inside a harness, one that no annotation could see coming. It is now a
static property of the type: a consumer that needs notes says ``Dataset[NoteAnnotated]``,
and a corpus without them cannot be passed to it.

**References are lazy.** ``reference()`` and ``reference_notes()`` are methods rather
than fields because building one can cost a MIDI parse per item, and a run that stops
early or filters most items out should not pay for all of them up front.

**Grouping is the item's business.** An item declares its own ``groups`` ("string 1",
"plucking FS", "split test") and the report machinery slices by whatever keys it finds.
That is how per-string and per-technique breakdowns survive without the evaluation layer
knowing anything about bass guitars.
"""

from collections.abc import Iterator, Mapping
from typing import Protocol, TypeVar, runtime_checkable

from stemify.audio.sources import Sources
from stemify.music.contour import Contour
from stemify.music.notes import NoteEvent


@runtime_checkable
class AudioItem(Protocol):
    """One track's identity and the audio it is about. Carries no ground truth."""

    @property
    def name(self) -> str:
        """Identifier, unique within its dataset. Shown in per-item breakdowns."""
        ...

    @property
    def groups(self) -> Mapping[str, str]:
        """Labels this item can be broken down by, e.g. ``{"string": "1"}``.

        A dataset with no meaningful strata returns an empty mapping and simply reports
        its aggregate. Required rather than defaulted: all four corpora implement it,
        so the default was dead code that only made the protocol longer.
        """
        ...

    def sources(self) -> Sources:
        """Every audio role this item can offer.

        One member rather than two. There used to be an ``audio()`` beside this,
        answering "which single file is this item *about*" against this one's "what else
        is on hand for the same music" -- but ``Sources.bass`` is defined as the audio
        that gets transcribed, so the two answers were the same answer and every caller
        had to know which spelling a given site had picked. Where a loader still needs
        the distinction internally (MDB selects stem-or-mix and raises if the mix is
        absent) it keeps a private helper; that is an implementation detail, not
        contract.
        """
        ...


@runtime_checkable
class FrameAnnotated(AudioItem, Protocol):
    """An item with frame-level pitch truth. Every corpus here satisfies this."""

    def reference_contour(self) -> Contour:
        """Ground-truth pitch over time, on the corpus's own grid.

        Named for what it returns, so it reads beside ``reference_notes`` as the pair it
        is. It used to be ``reference()``, which said *that* it was truth but not
        *which kind* -- and it was the one required member while the note-level truth
        every trainer actually reads was optional and raised.
        """
        ...


@runtime_checkable
class NoteAnnotated(FrameAnnotated, Protocol):
    """An item whose truth is note events, not only a contour.

    Slakh (rendered from MIDI) and FiloBass (transcribed to it) qualify;
    MDB-bass-synth and IDMT are frame-annotated and do not. Everything that trains or
    scores at the note level -- ``semicrf.fit.tracks``, ``training.validation``,
    ``evaluation.note`` -- takes this rather than ``AudioItem``, which is what turns
    "this corpus has no note boundaries" from a mid-run exception into a type error.

    ``runtime_checkable`` so the one consumer that can *degrade* rather than refuse
    (``evaluation.cues`` drops its onset channel) can ask at runtime. Presence is all
    that check verifies, which is exactly right here: the member either exists or the
    corpus does not have notes.
    """

    def reference_notes(self) -> list[NoteEvent]:
        """Ground-truth note events, for note-level training and evaluation."""
        ...


ItemT_co = TypeVar("ItemT_co", bound=AudioItem, covariant=True)


class Dataset(Protocol[ItemT_co]):
    """An iterable of evaluable items, owning how its references are built.

    Generic in the item so a consumer's requirement travels with the dataset it is
    handed: ``Dataset[NoteAnnotated]`` is a corpus whose items have note truth, and
    ``SlakhDataset`` satisfies it while ``MdbDataset`` does not. Covariant because a
    dataset only ever produces items, never consumes them.
    """

    @property
    def name(self) -> str:
        """Human-readable identity, including any variant -- the source or split matters
        as much as the dataset name when reading a result months later."""
        ...

    def __iter__(self) -> Iterator[ItemT_co]:
        ...

    def __len__(self) -> int:
        ...
