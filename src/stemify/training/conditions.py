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

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from stemify.cache import Cache
from stemify.data.base import Dataset as SlakhLikeDataset, EvalItem
from stemify.data.slakh import SlakhDataset
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

logger = logging.getLogger(__name__)


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


def _mono(name: str, source: str, transform: AudioTransform) -> Condition:
    """A single-channel condition -- the model sees one rendering, as before."""
    return Condition(name, source, (Channel("bass", transform),))


# A condition's tag becomes a path segment in every metric it produces
# (``note/<tag>/f1``), and MLflow accepts only these characters in a metric name.
# Checked at import because the alternative is what happened with a "+" separator:
# training ran to completion, exited 0, and every validation metric was rejected
# one epoch at a time inside an async callback that swallowed the error.
_METRIC_SAFE = re.compile(r"^[A-Za-z0-9_.:-]+$")


# The target is always the clean stem's MIDI regardless of which audio the features
# come from -- that asymmetry is the domain randomisation, and it is why the separated
# and mixture conditions are trainable at all.
#
# A tag encodes two independent axes, so it can be read without a lookup:
#
#   what is transcribed -- clean (bass stem) | separated (demucs) | mixture (raw)
#   what the model sees -- one channel, or `bichannel-` = that audio PLUS the mixture
#                          alongside it as a context channel
#
# The old names conflated those: `mix` meant "transcribe the raw mixture", while the
# `-mix` suffix in `stem-mix` meant "the mixture is a *context* channel" -- the same
# word for opposite roles. See LEGACY_NAMES for the mapping.
CONDITIONS: dict[str, Condition] = {
    # -- one channel ---------------------------------------------------------------
    # The pinned default: clean bass, single-bass tracks only (docs/ablation-plan.md).
    "clean": Condition(
        "clean", "stem", (Channel("bass", Identity()),), solo_only=True
    ),
    # The deployment condition -- what a user's upload becomes after separation.
    "separated": _mono("separated", "mix", DemucsSeparator()),
    # Negative control: no separation at all. Scores ~0.27, which is what makes
    # separation non-negotiable rather than a preference.
    "mixture": _mono("mixture", "mix", Identity()),
    # Historical only: the one condition that keeps multi-bass tracks, which is what
    # tcn-v1/tcn-v2 reported. Kept so those numbers stay reproducible by tag, but it is
    # *not* comparable to anything else here -- multi-bass tracks are strictly harder,
    # and mistaking that for a model difference is the error docs/ablation-plan.md
    # exists to stop.
    "clean-multibass": _mono("clean-multibass", "stem", Identity()),
    # -- two channels: the audio to transcribe, plus the mixture it came from --------
    # `bichannel-separated` is the deployment-shaped one; `bichannel-clean` is its
    # clean-bass control, separating what the mixture channel adds from what
    # separation damage costs.
    "bichannel-clean": Condition(
        "bichannel-clean", "stem",
        (Channel("bass", Identity()), Channel("mix", Identity())),
    ),
    "bichannel-separated": Condition(
        "bichannel-separated", "mix",
        (Channel("bass", DemucsSeparator()), Channel("mix", Identity())),
    ),
}

# Renamed 2026-07-30. Old tags still resolve (with a warning) so the run tags recorded
# in docs/experiments.md and any saved scripts keep working -- a rename that silently
# invalidates every documented experiment is worse than the names it fixes.
LEGACY_NAMES: dict[str, str] = {
    "stem-solo": "clean",
    "stem": "clean-multibass",
    "demucs": "separated",
    "mix": "mixture",
    "stem-mix": "bichannel-clean",
    "demucs-mix": "bichannel-separated",
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
CLEAN_BY_WIDTH: dict[int, str] = {1: "clean", 2: "bichannel-clean"}

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


def canonical(name: str) -> str:
    """A condition tag, with any retired name mapped to its current one.

    The single place the legacy mapping is applied. It exists because it was *not*
    single: ``resolve`` translated ``--sources`` while ``train`` looked up
    ``--val-sources`` straight in ``CONDITIONS``, so a run with old tags resolved its
    training conditions and then died on a ``KeyError`` for the val ones -- three
    seconds in, after the previous arm had already succeeded with the same arguments
    under the pre-rename code.

    Unknown names pass through untouched so the caller can raise with its own message.
    """
    if name in LEGACY_NAMES:
        logger.warning(
            "condition %r was renamed to %r; the old name still works but the metrics "
            "will be logged under the new one", name, LEGACY_NAMES[name],
        )
        return LEGACY_NAMES[name]
    return name


def resolve(names: list[str]) -> list[Condition]:
    """Condition tags -> conditions, rejecting a set the model cannot have one shape for.

    Mixing widths is the failure worth catching here: ``--sources clean,bichannel-separated``
    names one 1-channel and one 2-channel condition, and concatenating them yields
    batches whose channel count depends on which track was drawn. That surfaces
    deep inside the first conv as a shape error, or -- worse, if the loader happens
    to draw a uniform first batch -- not until some later epoch.
    """
    resolved = [canonical(n) for n in names]
    unknown = [n for n in resolved if n not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}; have {sorted(CONDITIONS)}"
        )
    conditions = [CONDITIONS[n] for n in resolved]
    widths = {c.in_channels for c in conditions}
    if len(widths) > 1:
        shapes = ", ".join(f"{c.name}={c.in_channels}ch" for c in conditions)
        raise ValueError(
            f"conditions must agree on input channel count, got {shapes}. One model "
            "has one input shape, so train the widths as separate runs."
        )
    return conditions
