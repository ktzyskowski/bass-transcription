"""``Reference``: one track's ground truth, on one grid, as one object.

A track's truth is *notes* and *where the bass is audible*. Those are not two facts --
the second is what the first sounds like -- and every consumer needs both. The frame
objective reads them as per-frame targets; the semi-CRF reads them as a segmentation.
**Two views of one reference.**

They were not held that way, and the cost was two silent disagreements that survived
into training:

``gold_segments`` derived its own answer to *"when does a note stop sounding"* from a
flat ``GAP_REST_S = 120 ms`` while never seeing the audio, and the voicing target
derived a different one by measuring each note's decay to ``-40 dB`` from its own attack
peak. On 40 real tracks that put **28% of gold's REST frames on frames the voicing
target calls voiced** -- and since the voicing head's output *is* the DP's rest
evidence, 55-59% of gold's rest spans then scored better merged into the note before
them. The objective was paying a median 8.8 nats a span to under-segment.

The second was smaller and the same shape: onsets were placed on the grid with
``searchsorted`` here and ``nearest_frame`` there, so the two arms disagreed by up to a
frame about when a note starts.

Neither was a wrong formula. Both were the *same* formula written twice. This type is
where it gets written once.

**Invariant: everything is on ``frame_times``.** A reference is only meaningful on a
grid, and the two arms run on different ones (the encoder's hop, and the decimated hop
the DP scans) -- so a reference carries its grid and ``decimated`` produces the other
view rather than callers slicing fields apart and hoping they stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stemify.music.notes import NoteEvent


@dataclass(frozen=True)
class Reference:
    """One track's notes and voiced mask, both on ``frame_times``."""

    frame_times: np.ndarray
    """The grid everything here is indexed by, in absolute seconds."""

    voiced: np.ndarray
    """``(T,)`` bool -- where the bass is *audible*, note decays included.

    Built by ``corpus.midi_contour.notes_to_contour``'s gate and carried through the
    cached ``voicing`` target, so it is the same signal the voicing head is trained
    against. That identity is the point: a segmentation labelling a frame ``rest`` while
    this says the bass is still ringing is asking the DP to score a claim its own
    evidence contradicts.
    """

    notes: tuple[NoteEvent, ...]
    """Reference note events, in onset order. The *events*, not their sound."""

    def __post_init__(self) -> None:
        if self.voiced.shape != self.frame_times.shape:
            raise ValueError(
                f"voiced has shape {self.voiced.shape} against a "
                f"{self.frame_times.shape} grid; a reference is only meaningful on one "
                "grid, and a length mismatch here silently shifts every label against "
                "the audio it describes"
            )
        if self.voiced.dtype != np.bool_:
            raise ValueError(
                f"voiced must be bool, got {self.voiced.dtype}. A soft mask has a "
                "threshold in it, and a threshold chosen at each call site is how the "
                "two arms came to disagree in the first place -- pick it once, in "
                "`from_targets`."
            )

    @property
    def n_frames(self) -> int:
        return int(self.frame_times.size)

    @classmethod
    def from_targets(
        cls,
        frame_times: np.ndarray,
        voicing: np.ndarray,
        notes: list[NoteEvent],
        threshold: float = 0.5,
    ) -> "Reference":
        """From a cached per-frame ``voicing`` target and the corpus's notes.

        **The one place the voicing target becomes a hard mask.** It arrives as a
        resampled float because ``resample_mask`` averages onto the feature grid, and
        the boundary frames are therefore fractional. Thresholding at each call site
        would be one more definition held in several places.
        """
        return cls(
            frame_times=np.asarray(frame_times),
            voiced=np.asarray(voicing) > threshold,
            notes=tuple(sorted(notes, key=lambda n: n.onset)),
        )

    def decimated(self, stride: int) -> "Reference":
        """This reference on the DP's grid -- every ``stride``-th frame.

        The notes ride through untouched: they are events in *seconds* and owe nothing
        to a grid. Only the mask and the clock are decimated, together, here.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        return Reference(
            frame_times=np.ascontiguousarray(self.frame_times[::stride]),
            voiced=np.ascontiguousarray(self.voiced[::stride]),
            notes=self.notes,
        )
