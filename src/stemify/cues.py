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

``onset``/``voicing``/``pitch_logits`` are per-frame on the ``frame_times`` grid (the
feature grid, ~11.6 ms); ``contour`` is a pitch track on its own grid. A decoded
note's pitch always comes from ``contour`` -- a decoder places, splits and merges
notes, but never invents a pitch, so a frame the model calls voiced yet the contour
left pitchless contributes no pitch rather than a guessed one.

**The contour and the logits are not redundant.** The contour is a point estimate,
one reading of a distribution; ``pitch_logits`` is the distribution. A segmental
decoder scores properties the point estimate cannot express -- how far pitch travelled
inside a span, how much the span's frames disagree -- so both travel together.
"""

from dataclasses import dataclass

import numpy as np

from stemify.contour import Contour


@dataclass(frozen=True)
class Cues:
    """Per-frame onset/voicing evidence plus the pitch contour, for a whole track."""

    # Pitch track on its own grid -- what names a decoded note. **Not** optional, unlike
    # the training-side ``TensorCues``: the DP scores spans with no contour at all, but
    # an inference ``Cues`` exists to be turned into ``NoteEvent``s, and a note needs a
    # pitch. Both encoders already guarantee one (``encode_features`` takes a required
    # ``Contour``), so the union only ever described a state no producer could reach --
    # while forcing every consumer that names a pitch to guard against it.
    contour: Contour
    onset: np.ndarray             # (T,) per-frame onset evidence (prob or DSP envelope)
    frame_times: np.ndarray       # (T,) feature-grid times, seconds
    voicing: np.ndarray | None = None  # (T,) per-frame P(voiced), or None (no head)
    # (T, n_bins) raw per-bin logits from a FrequencyHead, or None when the model has
    # no pitch head. Raw rather than softmaxed: the span features that read these
    # normalise across the bin axis themselves.
    #
    # This field is why the *decoded* result matches what was trained. The semi-CRF's
    # `path_length` and `pitch_spread` potentials are functions of these logits, and a
    # cues bundle that drops them makes both read zero -- so a decoder would score
    # spans as though w_path and w_spread were zero, however they were fitted. That is
    # not a degradation the loss can see, because it only happens after training.
    pitch_logits: np.ndarray | None = None

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
