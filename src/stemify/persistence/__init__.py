"""Bytes on disk: where an artifact lives, and what it becomes.

Two modules, and the split between them is the pillar's whole idea.

    store  *where* -- the directory layout and the content key. Knows nothing about
           what it is addressing.
    serde  *what bytes* -- ``CachedItem``'s split npz/npy layout, and the ``Contour``
           format. Knows nothing about when to write one.

Neither knows *whether to compute*, and that is the change this pillar records. It
replaced a ``Cache`` class whose central method walked a corpus, resolved an audio
transform, ran a CQT, wrote an npz and returned paths -- five concerns welded into one
call, which is why it imported all five layers and why every training test had to
subclass it into a ``_NoOpCache`` to get it out of the way. A pillar its own consumers
must stub is a pillar whose seam is in the wrong place.

Inverted, the memoisers live beside the things they memoise: ``audio.cache`` owns the
CQT. It is a handful of lines that asks the
store for a path, returns it on a hit, and computes on a miss. Adding a cached artifact
is a new memoiser next to its producer rather than a new method here.

**Every write is atomic** (``platform.atomic``), because each of these is a memoised
expensive computation and a truncated file is a *cache hit* on the next run. That
failure mode is silent by construction: not a crash, but a model trained on a
half-written CQT.

Below it: ``music`` for ``Contour``, and ``platform`` for the atomic write. Nothing
about audio, models or corpora -- which is what lets ``audio`` sit above it and hand it
a CQT to hold.
"""
