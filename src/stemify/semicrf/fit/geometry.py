"""How a track was cut into chunks, and every conversion that follows from it.

Six coordinate systems meet in the chunked semi-CRF, and until this module they were
all bare ``int``:

  1. **CQT frame** -- the model's grid, and what the cached features are indexed by.
  2. **DP frame** -- the ``STRIDE``-decimated grid the scan runs on. ``cqt = dp * stride``.
  3. **Chunk-local DP frame** -- a window's own coordinates, origin at its first
     context frame. Gold is rebased into this on its way out of the dataset.
  4. **Window slot** -- ``0 .. dmax-1``, the rolling alpha buffer's index.
  5. **Window offset** -- the recurrence's ``k``, where ``position = j - dmax + k``.
  6. **Duration** -- ``j - i``, which indexes the duration table.

Nothing distinguished them. A gold segment is ``(label, i, j)`` before it is rebased
and ``(label, i, j)`` after -- the same tuple type, a different frame of reference, no
marker either way. Both of the recorded bugs in this area came through that seam: a
chunk seeded at ``START`` when gold's own boundary was meant (5.18 nats/track, hidden
from every telescoping test by zero-initialised transitions), and a seed slot formula
that lived at its call site while only the range check lived beside the buffer.

This module is the conversions and nothing else. It does not *tag* values -- a
``localise`` still returns an ``int``, and nothing stops that int being used as a track
frame. What it buys is that each formula exists **once**, in the object that holds the
origin it needs, so two call sites cannot spell the same conversion differently. The
codebase already reached for this once, in ``forward.candidate_starts``: shared with
``viterbi`` on the stated grounds that *"deriving it twice is how an off-by-one becomes
a decode that is wrong only near track starts"*.

**Cold path only, and that is not a compromise.** The scan's per-step arithmetic
(``candidate_starts``) stays entirely within the scan's own frame -- ``j`` in, start
positions in the same coordinates out. Nothing crosses a frame boundary inside the
loop. Every conversion here happens once per chunk (rebasing), once per batch
(seeding), or once after the scan (readout), so a Python object on this path costs
nothing measurable, and the DP stays raw tensors where it is launch-bound.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkGeometry:
    """How the dataset cut a track, in DP frames. Uniform for every chunk it emits.

    Held on the batch rather than copied onto each chunk: these are properties of the
    *cutting*, identical across rows, and copying a constant onto every row only to
    check the copies still agree invents the invariant it then guards.
    """

    context: int
    """DP frames of left context before a chunk's own first frame.

    Equal to ``dmax`` today, and reaching that value by a different route -- the
    dataset takes it from ``SemiCRFRecipe._geometry``. The conversions below do not
    *require* them to be equal, so the two staying separate fields is honest rather
    than redundant: left context is "how far back the scan may look for a span start",
    and ``dmax`` is "the widest span the duration table can score". They coincide
    because the first is chosen to cover the second.
    """

    chunk: int
    """The chunk's own DP frames -- the range the NLL actually scores."""

    right: int
    """DP frames past the chunk that are read as context and never scored.

    The encoder's receptive field reaches forward, so the chunk's right edge needs
    real audio after it or its last frames see zero-padding the deployed model never
    would.
    """

    stride: int
    """DP frames to CQT frames. The scan runs decimated; the features do not."""

    @property
    def width(self) -> int:
        """A window's total DP frames. Every chunk is this wide -- uniform shape is
        what lets the scan be captured as a CUDA graph."""
        return self.context + self.chunk + self.right

    def cqt(self, dp_frames: int) -> int:
        """DP frames to CQT frames."""
        return dp_frames * self.stride

    def scored_slice(self, window_cqt_frames: int) -> tuple[int, int]:
        """The half-open CQT range the NLL scores: the chunk's own frames.

        One place, because every auxiliary term must use *this* range and nothing else.
        A term averaging over the whole window would divide by a different frame count
        than the NLL does, so the shared "nats per frame" unit -- the only thing that
        makes the head weights interpretable -- would quietly stop holding between
        terms. It would also supervise the margins twice per epoch, since one chunk's
        context is the previous chunk's scored region.
        """
        return self.cqt(self.context), window_cqt_frames - self.cqt(self.right)

    def seed_slot(self, local_start: int, dmax: int) -> int:
        """Where a gold segment starting at ``local_start`` sits in the alpha window.

        The scan's first step is ``j = context + 1`` and it reads window index ``k`` as
        position ``j - dmax + k``; inverting that gives this. A position equal to
        ``context`` -- "no frames consumed yet" -- lands on the last slot, which is why
        a track's first chunk needs no special case.

        Correct for any ``context``/``dmax`` pair, not only the equal one they take
        today. What it does require is that the segment start not precede the scan by
        more than ``dmax``; ``window.initial_window`` refuses the rest.
        """
        return local_start - self.context - 1 + dmax

    def at(self, origin: int) -> "ChunkFrame":
        """This geometry anchored at one chunk's window origin."""
        return ChunkFrame(self, origin)


@dataclass(frozen=True)
class ChunkFrame:
    """One chunk's window: its geometry, plus where its slot 0 sits in the track.

    Separate from ``ChunkGeometry`` because it needs an origin and the geometry does
    not, and that difference is exactly which conversions each can answer. A batch
    mixes chunks from different tracks at different offsets, so it can hold a geometry
    but never a frame; the dataset, cutting one track, holds a frame per chunk.
    """

    geometry: ChunkGeometry
    origin: int
    """Track DP frame of this window's first (context) frame. Negative on a track's
    first chunk, where the window begins before the track does and is zero-padded."""

    def localise(self, track_dp_frame: int) -> int:
        """A track DP frame in this window's own coordinates."""
        return track_dp_frame - self.origin

    def to_track(self, local_dp_frame: int) -> int:
        """The inverse of ``localise``."""
        return local_dp_frame + self.origin

    def cqt_window(self) -> tuple[int, int]:
        """``(start, width)`` in CQT frames -- the slice to read features from.

        The one place the DP grid becomes the feature grid for this window. Reading it
        with a start derived here and a width derived elsewhere is how a window ends up
        off by a stride, which shifts every target against its features.
        """
        return self.geometry.cqt(self.origin), self.geometry.cqt(self.geometry.width)
