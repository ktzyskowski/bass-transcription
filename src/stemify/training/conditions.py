"""A *condition*: which audio each of the model's input channels reads.

The old ``SOURCES`` table mapped a tag to one ``(Slakh source, transform)`` pair,
because a model had exactly one input. That assumption is what this replaces: a
condition now names an *ordered list* of channels, and a one-channel condition is
the degenerate case rather than the only one.

The motivating channel is the mixture. A separated bass stem is missing the very
evidence that disambiguates its own artifacts -- a low transient that lines up with
a kick in the mix is bleed, not a re-pluck, and separation leaves exactly that kind
of residue. The bass channel cannot tell those apart; the pair can. Since Slakh
ships ``mix.flac`` beside every stem and features are content-addressed per resolved
audio file, feeding it costs one extra cached CQT and no re-rendering.

**Channel 0 is load-bearing.** It carries the bass -- the audio the pitch contour is
tracked from and every per-frame target describes. The rest are context the encoder
may condition on but is never asked to transcribe. Nothing downstream re-derives
that ordering, so a condition that puts the mixture first would train a model to
transcribe the mixture with the bass as context, silently.

Conditions live here rather than in ``recipes.py`` because they answer a different
question -- "what does the model look at" against "what is it trained to optimise"
-- and they are what the *evaluation* harness varies too, where objectives do not
apply at all.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from stemify.cache import Cache
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.data.base import EvalItem
from stemify.data.slakh import SlakhDataset
from stemify.transforms import AudioTransform, DemucsSeparator, Identity


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

        For the three single-channel conditions this reproduces the previous
        ``SlakhDataset`` default exactly; what it adds is that ``stem-mix`` is
        filtered too, which the old ``source``-only rule would have missed -- its
        source is ``stem``, so nothing about it looked mix-flavoured.

        ``solo_only`` overrides the derivation. That exists for *controls*: a plain
        ``stem`` run keeps two-bass tracks (a stem is unambiguous whatever else the
        track contains, so they are free extra data), which means it and
        ``stem-mix`` train on different track sets -- at ``--limit 40``, 33 tracks
        against 37, sharing 26. Comparing them would vary the input channels and
        the corpus at once.
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


def _mono(name: str, source: str, transform: AudioTransform) -> Condition:
    """A single-channel condition -- the model sees one rendering, as before."""
    return Condition(name, source, (Channel("bass", transform),))


# A condition's tag becomes a path segment in every metric it produces
# (``note/<tag>/f1``), and MLflow accepts only these characters in a metric name.
# Checked at import because the alternative is what happened with a "+" separator:
# training ran to completion, exited 0, and every validation metric was rejected
# one epoch at a time inside an async callback that swallowed the error.
_METRIC_SAFE = re.compile(r"^[A-Za-z0-9_.:-]+$")


# The conditions a run can name. Single-channel entries reproduce the old
# ``SOURCES`` table exactly, so every existing result stays reproducible by tag.
#
# The target is always the clean stem's MIDI regardless of which audio the features
# come from -- that asymmetry is the domain randomisation, and it is why the mix and
# demucs conditions are trainable at all.
CONDITIONS: dict[str, Condition] = {
    "stem": _mono("stem", "stem", Identity()),
    "mix": _mono("mix", "mix", Identity()),
    "demucs": _mono("demucs", "mix", DemucsSeparator()),
    # Bichannel: the bass to transcribe, plus the mixture it was drawn from.
    # "demucs-mix" is the deployment-shaped one -- a separator's output is what a
    # user's upload actually becomes -- while "stem-mix" is its clean-bass control,
    # isolating what the mixture channel adds from what separation damage costs.
    "stem-mix": Condition(
        "stem-mix", "stem",
        (Channel("bass", Identity()), Channel("mix", Identity())),
    ),
    "demucs-mix": Condition(
        "demucs-mix", "mix",
        (Channel("bass", DemucsSeparator()), Channel("mix", Identity())),
    ),
    # The matched control for "stem-mix": one channel, but the same single-bass
    # track set, so an A/B against it varies the input channels and nothing else.
    # "demucs" needs no such twin -- it already filters to single-bass tracks, so
    # "demucs" vs "demucs-mix" is controlled as it stands.
    "stem-solo": Condition(
        "stem-solo", "stem", (Channel("bass", Identity()),), solo_only=True
    ),
}

# The clean-bass condition at each input width. A frame run validates on clean
# stems whatever it trains on, but it cannot feed a 2-channel model 1-channel
# windows -- so "the clean condition" depends on the width. Named here, next to the
# table, because a tag spelled inline in recipes.py is one a rename silently misses:
# that is exactly how ``val_loader`` came to reference a "stem+mix" that no longer
# existed, which no test caught because none builds a val loader without the corpus.
#
# Both entries are ``unambiguous_only`` (single-bass) deliberately. "stem" would be the
# obvious 1-channel choice and is wrong here: it is the *only* condition that keeps
# multi-bass tracks, so a 1-channel run would early-stop against a different, harder
# track set than its 2-channel counterpart -- which is exactly the mismatch that made
# tcn-v2's stem 0.878 look worse than stem-solo's 0.9375. See docs/ablation-plan.md.
CLEAN_BY_WIDTH: dict[int, str] = {1: "stem-solo", 2: "stem-mix"}

_bad_clean = {
    w: t for w, t in CLEAN_BY_WIDTH.items()
    if t not in CONDITIONS or CONDITIONS[t].in_channels != w
}
if _bad_clean:
    raise ValueError(f"CLEAN_BY_WIDTH entries do not match CONDITIONS: {_bad_clean}")
del _bad_clean

_unsafe = sorted(t for t in CONDITIONS if not _METRIC_SAFE.match(t))
if _unsafe:
    raise ValueError(
        f"condition tag(s) {_unsafe} cannot appear in a metric name; use only "
        "letters, digits, and _ . : -"
    )
del _unsafe


def resolve(names: list[str]) -> list[Condition]:
    """Condition tags -> conditions, rejecting a set the model cannot have one shape for.

    Mixing widths is the failure worth catching here: ``--sources stem,demucs-mix``
    names one 1-channel and one 2-channel condition, and concatenating them yields
    batches whose channel count depends on which track was drawn. That surfaces
    deep inside the first conv as a shape error, or -- worse, if the loader happens
    to draw a uniform first batch -- not until some later epoch.
    """
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}; have {sorted(CONDITIONS)}"
        )
    conditions = [CONDITIONS[n] for n in names]
    widths = {c.in_channels for c in conditions}
    if len(widths) > 1:
        shapes = ", ".join(f"{c.name}={c.in_channels}ch" for c in conditions)
        raise ValueError(
            f"conditions must agree on input channel count, got {shapes}. One model "
            "has one input shape, so train the widths as separate runs."
        )
    return conditions
