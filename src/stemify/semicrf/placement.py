"""Where a tensor lives, and the one-device rule the DP depends on.

The recurrence adds its terms together, so a batch whose evidence is split across
devices is a hard error the moment the encoder is not on CPU. These are the three
helpers that establish and check that invariant.

This module used to be ``tensor_cues.py`` and to own ``TensorCues`` -- the tensor twin
of the numpy ``Cues``, the same five fields written a second time. ``Frames`` holds
either, so the twin is gone and what remains is the placement logic that made the two
interoperate. That interoperation is the point and is still load-bearing:
``torch.as_tensor`` treats a numpy field as a constant and a tensor field as a graph
node, which is what lets one DP be differentiable under training and a constant fold at
decode.
"""

from __future__ import annotations

import numpy as np
import torch

from stemify.transcription.frames import Frames


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
    grad-bearing field survives the placement."""
    return torch.as_tensor(x).to(device)


def frames_device(frames: Frames) -> torch.device:
    """Where a bundle's evidence already lives: the encoder's device under training,
    CPU for numpy. Only a fallback -- callers that own a module (``SemiCRFDecoder``)
    pass its device explicitly instead."""
    onset = frames.onset
    if isinstance(onset, torch.Tensor):
        return onset.device
    return torch.device("cpu")
