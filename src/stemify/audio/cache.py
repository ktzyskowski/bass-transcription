"""The memoised CQT: the one derived artifact this layer owns.

Separate from ``features`` so that module stays pure DSP -- audio in, arrays out, no
disk. This is the get-or-compute half, and it lives here rather than in ``persistence``
because knowing that a CQT is expensive, that ``SIGNATURE`` identifies the producer, and
that a miss costs an audio load is *this* layer's knowledge. The store only answers
where the file goes.

**Why the transform is a parameter and not a resolved path.** The key is over
``transform.cache_path(source)`` -- where the transformed audio *would* be, computed
without producing it -- while the work uses ``transform.apply(source)``, which produces
it. Collapsing the two would make asking "is this track's CQT cached?" run Demucs first:
minutes of separation per cache **hit**, over a whole corpus. Both halves are this
layer's own (``audio.transforms``), so nothing is imported to get them.

It is per-file, not per-corpus. The predecessor took a whole ``Dataset`` plus a
transform plus a role and swept it, which is what forced a persistence object to import
the corpus layer. Sweeping is the caller's business -- it is a ``for`` loop, and the
callers already hold the dataset.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from stemify.audio.features import SIGNATURE, extract_features
from stemify.audio.transforms import AudioTransform
from stemify.persistence.serde import CachedItem
from stemify.persistence.store import Store

logger = logging.getLogger(__name__)


def feature_path(store: Store, resolved_audio: str | Path) -> Path:
    """Where the CQT for an **already transform-resolved** audio file is cached.

    The low-level key. Pure and cheap, so a caller can ask "is it cached?" without
    paying anything to find out.
    """
    return store.location("features", SIGNATURE, resolved_audio)


def ensure_feature(
    store: Store, transform: AudioTransform, source: str | Path
) -> Path:
    """The cached CQT for one track's audio, computing it only on a miss.

    Stored as a target-less ``CachedItem``: features plus the frame grid, no labels.
    Targets are layered on afterwards by whoever holds the reference, which is what lets
    one cache entry serve an onset-only run and a later run that adds voicing without
    recomputing a single CQT.

    The stored ``name`` is the resolved audio's stem rather than a corpus item's name,
    because the entry belongs to the *file*: two items over one audio file share it by
    construction. (Targets do not share that way -- see ``examples.ensure_targets``,
    which guards exactly that case.)
    """
    resolved = transform.cache_path(store, source)
    path = feature_path(store, resolved)
    if not path.exists():
        feats = extract_features(transform.apply(store, source))
        CachedItem(resolved.stem, feats.values, {}, feats.frame_times).save(path)
    return path


def ensure_features(
    store: Store,
    transform: AudioTransform,
    sources: Sequence[str | Path],
    label: str = "-",
) -> list[Path]:
    """``ensure_feature`` over many tracks, with a progress heartbeat.

    The loop is trivial; the logging is why this exists. A cold sweep is a separation
    plus a CQT per track over thousands of tracks, and without a heartbeat it is silent
    for hours. ``label`` names the channel being built so a second channel visibly races
    through an already-warm cache -- which is what a resumed run looks like, and worth
    seeing rather than inferring.
    """
    total, computed = len(sources), 0
    paths: list[Path] = []
    for i, source in enumerate(sources, start=1):
        path = feature_path(store, transform.cache_path(store, source))
        hit = path.exists()
        paths.append(ensure_feature(store, transform, source))
        computed += not hit
        if i % 25 == 0 or i == total:
            logger.info(
                "features[%s] %d/%d (%d computed, %d cached)",
                label, i, total, computed, i - computed,
            )
    return paths
