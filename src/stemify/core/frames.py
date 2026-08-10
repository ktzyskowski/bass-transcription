"""``Frames``: what an encoder hands a decoder -- a track's per-frame evidence.

The note pipeline is ``audio -> Encoder -> Frames -> Decoder -> notes``. This is the
intermediary, and it carries two different kinds of thing:

  - ``heads`` -- the *semantic* per-frame outputs, keyed by head name: an onset curve,
    a voicing curve, per-bin pitch logits. These are the same names ``CueNet`` emits
    and ``AUX_TERMS`` supervises, so one vocabulary spans network, loss and decoder,
    and a head that exists in one place cannot be spelled differently in another.
  - ``hidden`` -- the backbone's own per-frame representation, which every head is a
    function of. It says nothing semantically and is strictly more informative than
    all the heads together, so a decoder that pools learned features over spans reads
    *this*, while one that peak-picks a curve reads a head.

Separating those two is why this type replaced ``Cues``. ``Cues`` enumerated named
semantic fields, so a representation had nowhere to live, and adding a head meant
editing the type, its tensor twin, and every construction site.

**One type for training and inference.** The fields hold numpy at decode and torch
under training, and nothing here inspects which -- ``torch.as_tensor`` downstream
treats the first as a constant and the second as a graph node. That is what lets one
code path be differentiable in the one case and a constant fold in the other.
Previously ``Cues`` and ``TensorCues`` were the same five fields written twice, which
made "the same evidence" two types that had to be kept in step by hand.

**The evidence is soft.** Per-frame probabilities and logits, not thresholded onsets
or a voiced mask. That is what lets a decoder be more than greedy: ``GreedyDecoder``
peak-picks and thresholds, while ``SemiCRFDecoder`` scores the raw distributions and
recovers sub-threshold onsets the greedy pass discards.

**``contour`` is optional.** It is a pyin point estimate, and it is here because
``ClassicalEncoder`` has nothing else and a decoder still has to name a note. A model
carrying a ``FrequencyHead`` supersedes it outright, so it is no longer a required
field -- which is what it had become in practice, held by callers that never read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from stemify.music.contour import Contour

if TYPE_CHECKING:  # torch is not imported at runtime -- see below
    import torch

    Array: TypeAlias = np.ndarray | torch.Tensor
else:
    # This module is on the DSP path (``ClassicalEncoder`` reaches it), and that path
    # deliberately does not pull torch. Annotations are lazy under ``from __future__
    # import annotations``, so the union above is never evaluated at runtime and this
    # placeholder only exists for anything that reads the name eagerly.
    Array = np.ndarray

PROBABILITY_HEADS = frozenset({"onset", "voicing"})
"""Heads whose values travel as probabilities; everything else travels as raw logits.

A property of *this* type's contract, not of any network: it is what a consumer may
assume about ``heads[name]`` without asking who produced it. The binary heads are
squashed because their consumers are thresholds and peak-pickers (and because
``ClassicalEncoder`` emits a 0/1 spike train, which is a probability and not a logit),
while pitch stays raw because every consumer normalises across the bin axis itself and
squashing each bin independently first would be a different distribution.

The sigmoid that enforces this lives in ``models.frames``, beside the network whose
output it adapts -- not here, where it would need torch.
"""


@dataclass(frozen=True)
class Frames:
    """Per-frame evidence for one track, on one time grid."""

    heads: Mapping[str, Array]
    """Per-frame head outputs, ``(T,)`` or ``(T, k)``, keyed by head name.

    A mapping rather than fields because *which* heads exist is a property of the
    checkpoint, not of the pipeline: an onset-only model, an onset+voicing model and a
    model with a ``FrequencyHead`` all produce a valid ``Frames``, and asking which one
    you have is ``"pitch" in frames.heads`` rather than a ``None`` check against a field
    that only some producers fill.
    """

    frame_times: np.ndarray
    """``(T,)`` seconds. Always numpy: it is the grid, never differentiated."""

    hidden: Array | None = None
    """``(T, C)`` backbone activations, or ``None`` from a producer that has no network.

    Optional rather than required because ``ClassicalEncoder`` has no backbone at all,
    and because every checkpoint predating this field decodes without one. A decoder
    that needs it should say so when it is missing rather than pooling zeros.
    """

    contour: Contour | None = None
    """A pitch track on its own grid, for naming a decoded note. See the module docstring."""

    def __post_init__(self) -> None:
        n = len(self.frame_times)
        for name, value in self.heads.items():
            if value is None:
                raise ValueError(
                    f"head {name!r} is present and None. A head this producer does not "
                    "have is absent from the mapping -- ``'voicing' in frames.heads`` "
                    "is the question consumers ask, and a None entry answers it wrong "
                    "while every shape check downstream still passes."
                )
            if not 1 <= len(value.shape) <= 2 or value.shape[0] != n:
                raise ValueError(
                    f"head {name!r} has shape {tuple(value.shape)}; a per-frame head is "
                    f"(T,) or (T, k) with T={n} to match frame_times"
                )
        if self.hidden is not None and self.hidden.shape[0] != n:
            raise ValueError(
                f"hidden has shape {tuple(self.hidden.shape)}; its leading axis must be "
                f"T={n} to match frame_times"
            )

    @property
    def T(self) -> int:
        return len(self.frame_times)

    # --- the heads that exist today, named ------------------------------------------
    #
    # Accessors rather than fields: they document the shape and the units of the three
    # heads this project trains, and they keep every read site free of string literals,
    # while `heads` stays open so a fourth head costs nothing.

    @property
    def onset(self) -> Array:
        """``(T,)`` P(onset). Required -- a decoder with no onset evidence cannot place
        a note boundary at all, so its absence is an error rather than a ``None``."""
        try:
            return self.heads["onset"]
        except KeyError:
            raise ValueError(
                f"these Frames carry no onset head (have: {sorted(self.heads)}). "
                "Every encoder produces one; a bundle without it was built by hand."
            ) from None

    @property
    def voicing(self) -> Array | None:
        """``(T,)`` P(voiced), or ``None`` from a model with no voicing head -- in which
        case a consumer falls back to the contour's own ``freqs > 0``."""
        return self.heads.get("voicing")

    @property
    def pitch_logits(self) -> Array | None:
        """``(T, n_bins)`` raw per-bin logits, or ``None`` with no ``FrequencyHead``.

        Raw, per ``PROBABILITY_HEADS``. This is the field that keeps a decoded result
        matching what was trained: the semi-CRF's span potentials are functions of these
        logits, and a bundle that drops them makes those potentials read zero -- which
        no loss can see, because it only happens after training.
        """
        return self.heads.get("pitch")
