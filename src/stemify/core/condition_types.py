"""``Channel`` and ``Condition``: what a model looks at, as types.

Split out of ``conditions.py``, which held these plus the registry of concrete
conditions plus the lookup helpers. The types change when the *shape* of an input
changes -- a new role, a new transform hook; the registry changes whenever an
experiment wants a new combination, which is far more often.

One-way by construction: this module knows nothing about which conditions exist, so
``conditions.py`` can import it to build the table without a cycle. ``Channel`` and
``Condition`` are re-exported there, so existing imports are unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

from stemify.core.cache import Cache
from stemify.core.data.base import Dataset as SlakhLikeDataset, EvalItem
from stemify.core.data.slakh import SlakhDataset
from stemify.audio.transforms import AudioTransform

@dataclass(frozen=True)
class Channel:
    """One model input channel: a track audio role, seen through one transform.

    The pair is the unit because either alone is ambiguous -- ``demucs`` and the
    raw-mix channel both read a track's ``mix.flac`` and differ *only* in the
    transform, which is exactly what puts them on separate cache keys.
    """

    role: str                    # a ``Sources`` field: "bass" (transcribed) or "mix"
    transform: AudioTransform

    def audio(self, item: EvalItem) -> Path:
        """The resolved audio file, running the transform if it is not yet cached.

        Use this when the file itself is needed (pyin wants something to listen to).
        Feature caching goes through ``Cache.ensure_features``, which keys on
        ``transform.cache_path`` instead so that a cache *hit* never pays for a
        separation just to discover it was a hit.
        """
        return Path(self.transform.apply(item.sources().role(self.role)))


@dataclass(frozen=True)
class Condition:
    """The ordered channels a model reads, over one filtered set of tracks."""

    name: str
    source: str                       # SlakhDataset source -- what ``audio()`` is
    channels: tuple[Channel, ...]
    solo_only: bool | None = None     # None = derive; see ``unambiguous_only``

    @property
    def in_channels(self) -> int:
        return len(self.channels)

    @property
    def primary(self) -> Channel:
        """Channel 0 -- the bass. The contour and every target describe *this*."""
        return self.channels[0]

    @property
    def unambiguous_only(self) -> bool:
        """Whether to keep only tracks with exactly one bass stem.

        Required the moment the mixture reaches the model, whether as the thing
        being transcribed (``source="mix"``) or as a context channel. With two
        basses sounding, one reference contour cannot describe what is in the
        mixture, so the model would be shown two basses and scored against one.

What it adds over the old ``source``-only rule is that ``bichannel-clean`` is
        filtered too: its source is the stem, so nothing about it looked
        mix-flavoured, yet it feeds the mixture straight to the model.

        ``solo_only`` overrides the derivation, and ``clean`` sets it. A stem is
        unambiguous whatever else the track contains, so an unfiltered clean run
        gets two-bass tracks as free extra data -- but then it trains on a
        *different corpus* than ``bichannel-clean`` (at ``--limit 40``: 33 tracks
        against 37, sharing 26), and an A/B between them would vary the input
        channels and the track set at once. That is exactly how tcn-v2's stem 0.878
        came to look worse than a later 0.9375 that was mostly an easier subset;
        ``clean-multibass`` keeps the unfiltered behaviour for reproducing those
        runs and is comparable to nothing else.
        """
        if self.solo_only is not None:
            return self.solo_only
        return self.source == "mix" or any(c.role == "mix" for c in self.channels)

    def dataset(
        self, root: str, split: str, limit: int | None = None
    ) -> SlakhDataset:
        """The track set this condition trains or scores on."""
        return SlakhDataset(
            root, split=split, source=self.source, limit=limit,
            unambiguous_only=self.unambiguous_only,
        )

    def channel_paths(
        self, dataset: SlakhLikeDataset, cache: Cache
    ) -> list[tuple[Path, ...]]:
        """Cached CQT paths per track, ordered by model input channel.

        One ``ensure_features`` pass per channel rather than per track, so each
        channel's progress log reads as one sweep, and the transposition at the end
        is what binds a track's channels together. ``strict=True`` because a length
        mismatch between channels would silently pair track *i*'s bass with track
        *j*'s mixture -- labels intact, inputs scrambled, loss still falling.
        """
        per_channel = [
            cache.ensure_features(dataset, c.transform, role=c.role)
            for c in self.channels
        ]
        return list(zip(*per_channel, strict=True))
