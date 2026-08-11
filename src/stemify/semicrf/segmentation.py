"""``Segmentation``: a tiling of frames into labelled spans, and which frame it is in.

One definition. There were three -- ``cost.Segmentation``, ``score.Segmentation`` and
``fit.dataset.Segmentation`` -- identical in shape, and only the last said which
coordinate system its indices were in ("window-local DP indices"). The other two were
the same tuple type meaning whatever the caller happened to hold.

That silence is the seam both recorded bugs in this area came through. A gold tiling is
built in **track** DP frames by ``gold.gold_segments`` and rebased into a **window's
own** frames on its way out of the dataset; before and after, it was the same
``list[tuple[int, int, int]]`` with nothing to tell them apart.

**The container is tagged, not the index.** Nobody passes a lone frame number across a
frame boundary -- they pass a whole tiling -- so the boundary worth checking is the
list, and one ``NewType`` per frame catches at every function signature what a hundred
scalar annotations would catch at the same places. It also sidesteps ``NewType``'s real
limit: a tagged ``int`` decays back to ``int`` under arithmetic, while a list is never
added to anything.

``Segmentation`` itself stays untagged and is what the *scoring* core takes. Scoring a
tiling against evidence is frame-agnostic -- the two only have to agree, and they do by
construction, since a chunk's evidence and its gold are cut together. A ``NewType`` is
assignable to the type it wraps, so both tagged forms pass to those functions and
neither passes to the other.
"""

from typing import NewType

Span = tuple[int, int, int]
"""``(label, start, end)`` -- half-open ``[start, end)`` in whatever frame holds it."""

Segmentation = list[Span]
"""A contiguous tiling, ordered by start. Frame-agnostic: what the scorers take."""

TrackSegmentation = NewType("TrackSegmentation", Segmentation)
"""Indices in track DP frames -- what ``gold.gold_segments`` produces, spanning a
whole track from frame 0."""

LocalSegmentation = NewType("LocalSegmentation", Segmentation)
"""Indices in one chunk window's own DP frames, origin at its first context frame.

What a ``Chunk`` carries and what the loss seeds from. Produced only by
``fit.geometry.ChunkFrame.localise_all``, which is the one crossing between the two.
"""
