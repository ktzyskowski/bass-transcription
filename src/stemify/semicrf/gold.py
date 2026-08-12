"""Build the gold segmentation the NLL scores against, from a track's ``Reference``.

Isolated from the DP because this is where the labelling rule lives, and it is pure
numpy, so it earns dense unit tests of its own.

**Gold is a *view* of the reference, not a second derivation of it.** It used to take
only the note list and answer "when does this note stop sounding" from a flat 120 ms
constant, while the voicing target answered the same question by measuring each note's
decay. On 40 real tracks that put 28% of gold's REST frames on frames the voicing target
called voiced -- and the voicing head's output *is* the DP's rest evidence, so gold was
being scored against a signal it contradicted. 55-59% of rest spans scored better merged
into the note before them, a median 8.8 nats each, which is a per-span bribe to
under-segment about 76 times a track.

So the rule below reads the reference's own mask. **A rest is where the bass is not
audible** -- not where the MIDI happens to have a gap. The two definitions cannot drift
again because there is now only one.
"""

import numpy as np

from stemify.music.reference import Reference
from stemify.music.grids import nearest_frame
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.semicrf.segmentation import TrackSegmentation


def gold_segments(
    reference: Reference,
    dmax: int,
    labels: LabelSet = BINARY,
) -> TrackSegmentation:
    """Tile ``[0, T)`` into ``(label, start, end)`` gold segments.

    Each note owns its own frames and then its **ring-out** -- the voiced frames after
    its MIDI offset, up to the next onset. What is left over, where the reference says
    the bass is silent, is a ``rest``.

    A note longer than ``dmax`` becomes consecutive segments of the *same* label rather
    than being truncated into a rest. The DP cannot represent it as one span, and a
    ``rest`` in the middle of a sounding note would teach silence the audio does not
    contain; consecutive same-pitch spans are what a unison transition already models.

    **Onsets are placed with ``nearest_frame``**, the same function the frame path's
    onset target uses. Rounding *up* to the next frame -- which is the right answer to
    "which frames does this note occupy" -- made every decoded onset late by up to a
    frame, since the decoder reads a note's time back off this index.

    **``labels`` is what makes a gold note name a pitch.** Under ``BINARY`` every note is
    label 0 and this is the two-label rule it always was. Under a pitch alphabet label 0
    is *the lowest representable semitone*, so defaulting the alphabet where a wide one
    is meant does not raise, does not change a shape, and trains every note in the
    corpus toward E1.
    """
    if dmax < 1:
        raise ValueError(
            f"dmax must be at least 1 frame, got {dmax}; a segment cap of zero admits "
            "no span that advances, so no tiling of a non-empty grid exists"
        )
    n_frames = reference.n_frames
    if n_frames == 0:
        return TrackSegmentation([])

    times, voiced = reference.frame_times, reference.voiced
    segs: list[tuple[int, int, int]] = []

    def emit(label: int, lo: int, hi: int) -> None:
        """Append ``[lo, hi)`` as consecutive spans no longer than ``dmax``.

        One helper for notes and rests alike: the cap is the DP's, not the label's, so
        expressing it once is what stops a long note becoming a rest in its own tail.
        """
        while lo < hi:
            capped = min(hi, lo + dmax)
            segs.append((label, lo, capped))
            lo = capped

    onsets = (
        np.clip(
            nearest_frame(times, np.array([n.onset for n in reference.notes])),
            0, n_frames - 1,
        ).tolist()
        if reference.notes else []
    )

    cursor = 0
    for k, note in enumerate(reference.notes):
        start = max(onsets[k], cursor)
        # A note owns frames up to the next onset; its ring-out lives in there. Past the
        # last note, up to the end of the grid.
        limit = onsets[k + 1] if k + 1 < len(reference.notes) else n_frames
        limit = max(limit, start)
        if start >= n_frames or limit <= start:
            continue
        if start > cursor:
            # Whatever precedes this note and is not some earlier note's ring-out.
            emit(labels.REST, cursor, start)
        # The note sounds while the reference says so. `nearest_frame` may put the
        # onset a frame before the first voiced frame, so the note always owns at
        # least its own first frame.
        sounding = np.flatnonzero(voiced[start:limit])
        end = start + 1 if sounding.size == 0 else start + int(sounding[-1]) + 1
        emit(labels.label_of(note.midi_pitch), start, end)
        cursor = end

    if cursor < n_frames:
        emit(labels.REST, cursor, n_frames)
    return TrackSegmentation(segs)
