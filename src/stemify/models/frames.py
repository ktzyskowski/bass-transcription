"""``CueNet``'s output dict -> ``Frames``: the one adapter both paths run through.

Training and inference used to build their own intermediary -- ``TensorCues`` in the
loss, ``Cues`` in ``LearnedEncoder`` -- from the same forward pass, with the squashing
and the axis handling written twice. They agreed by inspection, which is the weakest
form of agreement available: a head added to one and not the other is not a type error,
not a shape error, and not visible in a loss curve.

One function now, and ``to_numpy`` after it for the path that leaves the graph. Lives
in ``models`` rather than ``core`` because it knows two things about the network -- that
``FrameHead`` emits a trailing singleton axis, and which heads are probabilities -- and
because it needs torch, which the DSP path deliberately does not pull.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from stemify.core.contour import Contour
from stemify.core.frames import PROBABILITY_HEADS, Frames


def _per_frame(value: torch.Tensor) -> torch.Tensor:
    """``(B, T, 1) -> (B, T)`` for a scalar head, leaving a wide one alone.

    ``FrameHead`` emits ``out_dim`` on a trailing axis, so a scalar head arrives with a
    length-1 tail that every per-frame consumer would otherwise squeeze at each read. A
    ``FrequencyHead``'s bin axis is never 1, so the test does not catch pitch.
    """
    return value.squeeze(-1) if value.shape[-1] == 1 else value


def frames_from_output(
    out: Mapping[str, torch.Tensor],
    frame_times: Sequence[np.ndarray],
    hidden: torch.Tensor | None = None,
    contours: Sequence[Contour | None] | None = None,
) -> list[Frames]:
    """A batched forward pass -> one ``Frames`` per row, still on the autograd graph.

    ``out`` is ``{head: (B, T, k)}`` exactly as ``CueNet.forward`` returns it, and
    ``hidden`` is ``(B, T, C)`` when the caller tapped the backbone. Heads named in
    ``PROBABILITY_HEADS`` are squashed here and nowhere else; the rest pass through raw.

    Grad-bearing on purpose: this is how a structured loss over segments reaches both
    the heads and the trunk. The inference path calls ``to_numpy`` afterwards.
    """
    heads: dict[str, torch.Tensor] = {}
    for name, value in out.items():
        v = _per_frame(value)
        heads[name] = torch.sigmoid(v) if name in PROBABILITY_HEADS else v

    rows = len(frame_times)
    for name, v in heads.items():
        if v.shape[0] != rows:
            raise ValueError(
                f"head {name!r} has batch {v.shape[0]} but {rows} frame_times grids "
                "were given; one grid per row, or the rows silently pair with the "
                "wrong track's times"
            )
    return [
        Frames(
            heads={name: v[b] for name, v in heads.items()},
            frame_times=frame_times[b],
            hidden=None if hidden is None else hidden[b],
            contour=None if contours is None else contours[b],
        )
        for b in range(rows)
    ]


def frames_from_logits(
    frame_times: np.ndarray,
    *,
    onset: torch.Tensor,
    voicing: torch.Tensor | None = None,
    pitch: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    contour: Contour | None = None,
) -> Frames:
    """One track's raw head logits -> ``Frames``. The unbatched peer of the above.

    Keyword-only because the heads are interchangeable in type and not in meaning:
    positionally, ``onset`` and ``voicing`` are two ``(T,)`` float tensors, and swapping
    them is neither a type error nor a shape error -- it trains, and it decodes silence
    as notes. Delegates rather than repeating the squash, so there is still exactly one
    place that decides what ``heads["onset"]`` holds.
    """
    named = {"onset": onset, "voicing": voicing, "pitch": pitch}
    out = {name: t[None] for name, t in named.items() if t is not None}
    (frames,) = frames_from_output(
        out,
        [frame_times],
        hidden=None if hidden is None else hidden[None],
        contours=[contour],
    )
    return frames


def to_numpy(frames: Frames) -> Frames:
    """The same ``Frames`` off the graph and off the accelerator.

    For the inference path, which hands its result to numpy/scipy decoders and to a
    cache. Detaching is implicit in ``.cpu().numpy()``; it is spelled out because a
    tensor that reaches a report while still holding a graph pins the whole forward pass
    in memory for as long as the report lives.
    """
    def off(x: np.ndarray | torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    return Frames(
        heads={name: off(v) for name, v in frames.heads.items()},
        frame_times=frames.frame_times,
        hidden=None if frames.hidden is None else off(frames.hidden),
        contour=frames.contour,
    )
