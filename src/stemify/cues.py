"""``Cues``: the intermediary between an encoder and a decoder.

The note pipeline is ``audio -> Encoder -> Cues -> Decoder -> notes``. ``Cues`` is
what the encoder produces and the decoder consumes -- the per-frame evidence for
where notes are, paired with the pitch track that says what they are. It lives at
top level next to ``Contour`` and ``NoteEvent`` because, like them, it is a neutral
data type both sides of a stage exchange; keeping it here lets ``encoders`` and
``decoders`` stay independent leaf packages that never import each other.

The evidence is deliberately *soft* -- per-frame probabilities (a learned onset
curve, a voicing curve) or a DSP envelope, not thresholded onsets or a voiced mask.
That is the whole reason a decoder can be more than greedy: ``HeuristicDecoder``
peak-picks and thresholds these into discrete cues, but ``SemiCRFDecoder`` consumes
the raw probabilities, which is how it recovers sub-threshold onsets the heuristic
would throw away.

``onset``/``voicing`` are per-frame on the ``frame_times`` grid (the feature grid,
~11.6 ms); ``contour`` is the pyin pitch track on its own grid. Pitch always comes
from ``contour`` -- a decoder places, splits and merges notes, but never invents a
pitch, so a frame the model calls voiced yet pyin left pitchless contributes no
pitch rather than a guessed one.
"""

from dataclasses import dataclass

import numpy as np

from stemify.contour import Contour


@dataclass(frozen=True)
class Cues:
    """Per-frame onset/voicing evidence plus the pitch contour, for a whole track."""

    contour: Contour              # pyin pitch track (own grid) -- the pitch source
    onset: np.ndarray             # (T,) per-frame onset evidence (prob or DSP envelope)
    frame_times: np.ndarray       # (T,) feature-grid times, seconds
    voicing: np.ndarray | None = None  # (T,) per-frame P(voiced), or None (no head)

    def __post_init__(self) -> None:
        if self.onset.shape != self.frame_times.shape:
            raise ValueError(
                f"onset {self.onset.shape} must match frame_times "
                f"{self.frame_times.shape}"
            )
        if self.voicing is not None and self.voicing.shape != self.frame_times.shape:
            raise ValueError(
                f"voicing {self.voicing.shape} must match frame_times "
                f"{self.frame_times.shape}"
            )
