"""Build the gold segmentation the NLL scores against, from reference notes.

Isolated from the DP because the audible-gap rule -- deciding which between-note
gaps are real rests and which are note-off artifacts to close -- is where subtle
label bugs hide, and it is pure numpy, so it earns dense unit tests of its own.
"""

import numpy as np

from stemify.semicrf.constants import GAP_REST_S
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.semicrf.segmentation import TrackSegmentation
from stemify.music.notes import NoteEvent


def gold_segments(
    frame_times: np.ndarray,
    ref: list[NoteEvent],
    dmax: int,
    labels: LabelSet = BINARY,
) -> TrackSegmentation:
    """Tile ``[0, T)`` into ``(label, start, end)`` gold segments from reference notes.

    Each note is a ``note`` span (clipped to ``dmax``); the space between notes is a
    ``rest`` only when the gap is longer than ``GAP_REST_S`` -- real audible silence.
    A shorter gap is a MIDI note-off artifact the audio rings through, so the previous
    note is extended to close it, yielding contiguous ``note->note`` gold. Training on
    the raw MIDI gaps instead teaches rests the audio never contains and a spurious
    ``note->note`` penalty that collapses legato runs (the under-segmentation trap).

    **``labels`` is what makes a gold note name a pitch.** Under ``BINARY`` every note
    is label 0 and this is the two-label rule it always was. Under a pitch alphabet
    label 0 is *the lowest representable semitone*, so defaulting the alphabet where a
    wide one is meant does not raise, does not change a shape, and trains every note in
    the corpus toward E1."""
    n_frames = frame_times.size
    spans = []
    for note in sorted(ref, key=lambda n: n.onset):
        start = int(np.clip(
            np.searchsorted(frame_times, note.onset), 0, n_frames - 1
        ))
        end = int(np.clip(
            np.searchsorted(frame_times, note.offset), start + 1, n_frames
        ))
        spans.append((start, end, labels.label_of(note.midi_pitch)))

    segs: list[tuple[int, int, int]] = []

    def add_rest(lo: int, hi: int) -> None:
        while lo < hi:
            capped = min(hi, lo + dmax)
            segs.append((labels.REST, lo, capped))
            lo = capped

    cur = 0
    for start, end, label in spans:
        start = max(start, cur)
        if end <= start:
            continue
        if start > cur:  # a gap: audible rest, or a note-off artifact to close?
            gap_s = (
                float(frame_times[start] - frame_times[cur]) if cur < n_frames
                else GAP_REST_S + 1
            )
            if gap_s > GAP_REST_S:
                add_rest(cur, start)
            elif segs and labels.is_note(segs[-1][0]):  # extend prev over the artifact
                prev_start = segs[-1][1]
                prev_end = min(start, prev_start + dmax)
                # The *previous* note's label, not this one's: the extension closes a
                # note-off artifact by holding the note that was already sounding.
                segs[-1] = (segs[-1][0], prev_start, prev_end)
                if prev_end < start:  # dmax-limited: remainder must still tile
                    add_rest(prev_end, start)
            else:
                add_rest(cur, start)  # no prior note to extend
        # clip an over-long note to one representable segment
        end = min(end, start + dmax)
        segs.append((label, start, end))
        cur = end
    if cur < n_frames:
        add_rest(cur, n_frames)
    return TrackSegmentation(segs)
