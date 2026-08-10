"""Cached training examples: a track's CQT with the labels a run asked for.

One job, split from the two neighbours it used to be welded to. `audio.cache` owns the
expensive half (the CQT, memoised per resolved audio file) and `music.labels` owns the
pure half (target arrays from a reference and a note list). This is the part that is
neither: walk a corpus, pair each item with its cache entry, and back-fill whatever
targets the entry is missing.

    prepare_examples   corpus -> cached (features + targets), returning paths
    ensure_targets     the back-fill alone, for callers that already hold the paths

Both exist because the two training paths reach the same cache by different routes: the
frame objective sweeps a dataset, while the structured one resolves channels through
``Condition.channel_paths`` and needs only the labels layered on. The ownership guard in
``ensure_targets`` has to be shared by both, and is exactly the kind of rule that drifts
once it is copied.

**Why the split was worth making.** ``build_example`` used to return a ``CachedItem``,
which welded target construction to the on-disk container -- so testing the target
alignment meant naming a persistence type, and the only caller that wanted a whole
``CachedItem`` was the one writing a *target-less* one. The targets are now
``music.labels.build_targets``: arrays in, arrays out, no disk, no corpus, no librosa.
"""

from stemify.examples.prepare import ensure_targets, prepare_examples

__all__ = ["ensure_targets", "prepare_examples"]
