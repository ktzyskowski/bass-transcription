"""``TensorCues``: the training-side twin of ``Cues``, and the one-device rule.

``Cues`` (numpy) is what an encoder hands a decoder at inference. This is the same
five fields as tensors, so the DP's arithmetic stays in the autograd graph and a
gradient reaches the encoder. The two model one concept twice, which is a known and
still-open design question -- see the ``Cues`` protocol discussion.

Split out of ``evidence.py``: that module was 415 lines holding this, the strided
``Evidence`` the DP scans, the prefix-sum helpers, and the builder between them. Those
change on different occasions -- this one when the encoder's outputs change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues


def check_one_device(what: str, **fields: torch.Tensor) -> None:
    """Raise unless every named field shares a device. The guard behind the one-device
    invariant: it names the offending field, which a ``RuntimeError`` raised deep in
    the recurrence does not."""
    by_device: dict[torch.device, list[str]] = {}
    for name, t in fields.items():
        by_device.setdefault(t.device, []).append(name)
    if len(by_device) > 1:
        detail = ", ".join(
            f"{'/'.join(sorted(names))} on {dev}" for dev, names in by_device.items()
        )
        raise ValueError(f"{what} fields span multiple devices: {detail}")


def on_device(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """``x`` as a tensor on ``device``. ``as_tensor`` keeps a numpy input a constant
    and a tensor input on the autograd graph; ``.to`` is itself differentiable, so a
    grad-bearing cue survives the placement."""
    return torch.as_tensor(x).to(device)


def cues_device(cues: Cues | TensorCues) -> torch.device:
    """Where a cues bundle's evidence already lives: the encoder's device for a
    ``TensorCues``, CPU for a numpy ``Cues``. Only a fallback -- callers that own a
    module (``SemiCRFDecoder``) pass its device explicitly instead."""
    if isinstance(cues.onset, torch.Tensor):
        return cues.onset.device
    return torch.device("cpu")



@dataclass(frozen=True)
class TensorCues:
    """The tensor peer of ``Cues``: the training-time intermediary.

    Field-for-field a ``Cues``, but ``onset``/``voicing`` are grad-bearing tensors (the
    encoder's forward output) rather than detached numpy, so a loss over them
    backprops into the encoder. ``contour`` and ``frame_times`` stay numpy: they are
    constants (pyin's pitch, the time grid) that never need gradients. ``Cues`` remains
    the *inference* intermediary; this is only built inside a training step. It lives
    here, not beside ``Cues``, so ``cues.py`` stays torch-free and the light DSP path
    never pulls torch."""

    # Pitch track (numpy) -- used to name a decoded note's pitch, never scored by the
    # DP. None when the model's own pitch head supplies it instead.
    contour: Contour | None
    onset: torch.Tensor            # (T,) P(onset), differentiable
    frame_times: np.ndarray        # (T,) feature-grid times, seconds
    voicing: torch.Tensor | None = None  # (T,) P(voiced), differentiable, or None
    # (T, n_bins) raw per-bin logits from a FrequencyHead, or None when the model has
    # no pitch head. Raw rather than softmaxed, because the span features normalise
    # across the bin axis themselves and squashing twice would be the wrong
    # distribution. Grad-bearing: this is how the CRF objective reaches the pitch head.
    pitch_logits: torch.Tensor | None = None

    @classmethod
    def from_logits(
        cls,
        contour: Contour | None,
        onset_logit: torch.Tensor,
        frame_times: np.ndarray,
        voicing_logit: torch.Tensor | None = None,
        pitch_logits: torch.Tensor | None = None,
    ) -> TensorCues:
        """Wrap a model's raw head logits, squashing the *binary* heads to the
        probabilities ``Cues`` carries so the two intermediaries hold the same quantity
        in the same range. Pitch stays as logits -- see the field."""
        voicing = None if voicing_logit is None else torch.sigmoid(voicing_logit)
        return cls(
            contour, torch.sigmoid(onset_logit), frame_times, voicing, pitch_logits
        )

