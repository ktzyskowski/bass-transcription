"""Heads: a backbone's features -> the per-frame quantity one objective predicts.

Split into two modules along the line that actually matters -- **what tensor a head
consumes**:

  - ``frame`` -- onset and voicing. Per-frame *scalars*, so each is a plain
    ``nn.Linear`` over the backbone's ``(B, T, hidden)`` features. They differ only
    in their target and their loss weighting, so they live together and change
    together.
  - ``pitch`` -- a position along the *frequency* axis, which those features no
    longer have (``_ConvFrontend`` flattens it away). So the pitch branch taps the
    front-end before that flatten and stays 2-D, ``(B, C, F, T) -> (B, T, F)``.

That difference is not cosmetic: it is why ``CueNet`` holds ``pitch`` apart from
its ``heads`` ModuleDict rather than alongside the other two -- a dict of modules
taking one argument shape cannot hold both without lying about it.

Everything is re-exported here, so ``from stemify.models.heads import onset_head``
keeps working exactly as it did when this was a single module.
"""

from stemify.models.heads.frame import FrameHead, onset_head, voicing_head
from stemify.models.heads.pitch import (
    CONTOUR_VOICING_THRESHOLD,
    FrequencyHead,
    decode_bins,
    decode_contour,
    pitch_ce,
    pitch_head,
    two_hot,
)

__all__ = [
    "CONTOUR_VOICING_THRESHOLD",
    "FrameHead",
    "FrequencyHead",
    "decode_bins",
    "decode_contour",
    "onset_head",
    "pitch_ce",
    "pitch_head",
    "two_hot",
    "voicing_head",
]
