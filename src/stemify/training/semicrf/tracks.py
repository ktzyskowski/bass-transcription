"""One track's cached inputs, resolved once: which CQTs, which contour, which notes.

Both structured-training datasets -- whole tracks (``sequence``) and ordered chunks
(``chunked``) -- need the same three things per track and used to carry their own copy
of the resolution logic::

    transform = transform or Identity()
    cache.ensure_features(dataset, transform)
    for item in dataset:
        audio = transform.apply(item.audio())
        ... cache.feature_path(audio), cache.contour(estimator, audio)

That predates ``training/conditions.py``. It hardcodes a single ``(source, transform)``
pair, so neither dataset could feed a model more than one input channel, and the two
copies were free to drift. This is that resolution, once, expressed in terms of a
``Condition`` -- so a mix-conditioned semi-CRF is a condition tag rather than a
rewrite.

**Channel 0 is the bass.** It is the audio a contour would be tracked from and the
audio the reference notes describe; the rest are context the encoder may read. Same
invariant as the frame path, and nothing downstream re-derives it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stemify.cache import Cache
from stemify.contour import Contour
from stemify.data.base import Dataset as SlakhLikeDataset
from stemify.notes import NoteEvent
from stemify.pitch import PitchEstimator
from stemify.serde import TrackExample
from stemify.training.conditions import Condition


@dataclass(frozen=True)
class TrackInputs:
    """Everything a structured objective needs for one track, as paths and values.

    Holds feature *paths* rather than arrays so a full-scale set never sits in RAM and
    the object pickles cheaply across DataLoader workers -- the contour and notes are
    small enough to carry directly.
    """

    features: tuple[Path, ...]   # cached CQT per input channel; [0] is the bass
    # A DSP contour over channel 0, or None when nothing needs one. No DP potential
    # reads it -- see ``track_inputs``.
    contour: Contour | None
    notes: list[NoteEvent]       # reference notes, always the clean stem's truth

    def open(self, mmap: bool = True) -> tuple[TrackExample, ...]:
        """This track's cached examples, one per channel.

        ``mmap`` keeps the features on disk so a window is paged in per read and the
        OS page cache is shared across workers; pass ``False`` to materialise.
        """
        return tuple(TrackExample.load(p, mmap=mmap) for p in self.features)


def channel_window(
    examples: tuple[TrackExample, ...], start: int, width: int
) -> np.ndarray:
    """``(C, n_bins, width)`` of stacked channel features, zero-padded off the ends.

    ``start`` may be negative and ``start + width`` may run past the track: both are
    normal for a chunk's context margins, and the padding is what keeps every window
    the same width -- which is what lets chunks from streams at different positions
    share a batch and keeps the DP's shapes static.
    """
    n_bins, n_frames = examples[0].features.shape
    out = np.zeros((len(examples), n_bins, width), dtype=np.float32)
    lo, hi = max(0, start), min(n_frames, start + width)
    if lo < hi:
        for c, ex in enumerate(examples):
            out[c, :, lo - start : hi - start] = ex.features[:, lo:hi]
    return out


def track_inputs(
    dataset: SlakhLikeDataset,
    condition: Condition,
    cache: Cache,
    estimator: PitchEstimator | None = None,
) -> list[TrackInputs]:
    """Resolve and cache every track's inputs for one condition, in dataset order.

    The CQT of each channel is ensured through ``Condition.channel_paths`` (one
    ``ensure_features`` sweep per channel, so a second channel is typically a pure
    cache hit from whichever run already built that condition), and the contour comes
    from channel 0 only -- the context channels are never pitch-tracked.

    ``estimator`` is *injected*, and passing ``None`` means **no contour at all**
    rather than "default to pyin". No DP potential reads a contour: the scan scores
    onset, voicing, duration, transitions, and the pitch head's own path/spread. So
    running pyin over the corpus to train a semi-CRF buys nothing but the wall time,
    and the default is off. Supply an estimator only when something downstream needs a
    pitch track -- the voicing fallback for a model with no voicing head, or naming
    pitches at decode.
    """
    per_track = condition.channel_paths(dataset, cache)
    return [
        TrackInputs(
            features=paths,
            contour=(
                None if estimator is None
                else cache.contour(estimator, condition.primary.audio(item))
            ),
            notes=item.reference_notes(),
        )
        for item, paths in zip(dataset, per_track, strict=True)
    ]
