"""``Cues | TensorCues`` -> the per-frame tensors the DP scans.

``TensorCues`` is the training-time peer of ``Cues`` (grad-bearing onset/voicing so
the loss reaches the encoder). ``build_emissions`` decimates by ``STRIDE`` and turns
either intermediary into an ``Emissions``: an onset logit, prefix cumsums of the
voicing log-probs, and a prefix cumsum of the pyin pitch-change indicator. The
cumsums make a segment's mass an O(1) difference ``C[j] - C[i]`` rather than a re-sum
per candidate span -- what keeps the scan linear in the segment cap.

``torch.as_tensor`` is the load-bearing trick: it treats a numpy ``Cues`` field as a
constant (inference/decode) and a ``TensorCues`` tensor as a graph node (training),
so one builder serves both and the DP is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.grids import resample_mask

# How far pyin's pitch must move for ``pitch_jumps`` to mark a frame. This was
# ``heuristic.PITCH_SPLIT_SEMITONES`` until that constant was turned off by default
# (a pitch change no longer *places* a note there -- see its docstring). The two
# usages were never the same question: there the threshold decided a hard note
# boundary, here it only raises a feature the CRF is free to weight, so the value is
# pinned locally at what every recorded semi-CRF run used rather than following a
# default that now means something else.
JUMP_SEMITONES = 0.8


def _check_one_device(what: str, **fields: torch.Tensor) -> None:
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


def _on(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """``x`` as a tensor on ``device``. ``as_tensor`` keeps a numpy input a constant
    and a tensor input on the autograd graph; ``.to`` is itself differentiable, so a
    grad-bearing cue survives the placement."""
    return torch.as_tensor(x).to(device)


def _cues_device(cues: Cues | TensorCues) -> torch.device:
    """Where a cues bundle's evidence already lives: the encoder's device for a
    ``TensorCues``, CPU for a numpy ``Cues``. Only a fallback -- callers that own a
    module (``SemiCRFDecoder``) pass its device explicitly instead."""
    if isinstance(cues.onset, torch.Tensor):
        return cues.onset.device
    return torch.device("cpu")


@dataclass(frozen=True)
class TensorCues:
    """The tensor peer of ``Cues``: the training-time intermediary.

    Field-for-field a ``Cues``, but ``onset``/``voicing`` are grad-bearing tensors
    (the encoder's forward output) rather than detached numpy, so a loss over them
    -- ``SemiCRFDecoder.nll`` -- backprops into the encoder. ``contour`` and
    ``frame_times`` stay numpy: they are constants (pyin's pitch, the time grid) that
    never need gradients. ``Cues`` remains the *inference* intermediary; this is only
    built inside a training step, from the model's logits. It lives here, not beside
    ``Cues``, so ``cues.py`` stays torch-free and the light DSP path never pulls torch."""

    contour: Contour               # pyin pitch track (numpy) -- the pitch source
    onset: torch.Tensor            # (T,) P(onset), differentiable
    frame_times: np.ndarray        # (T,) feature-grid times, seconds
    voicing: torch.Tensor | None = None  # (T,) P(voiced), differentiable, or None

    @classmethod
    def from_logits(
        cls,
        contour: Contour,
        onset_logit: torch.Tensor,
        frame_times: np.ndarray,
        voicing_logit: torch.Tensor | None = None,
    ) -> TensorCues:
        """Wrap a model's raw head logits, squashing to the probabilities ``Cues``
        carries so the two intermediaries hold the same quantity in the same range."""
        voicing = None if voicing_logit is None else torch.sigmoid(voicing_logit)
        return cls(contour, torch.sigmoid(onset_logit), frame_times, voicing)


@dataclass
class Emissions:
    """The strided per-frame evidence for one track, as the DP consumes it.

    Voicing and the pitch-change cue enter as prefix cumsums so a segment's mass is
    an O(1) difference ``C[j] - C[i]`` rather than a re-sum per candidate span --
    what keeps the scan linear in the segment cap.

    **Device invariant: all four tensor fields live on one device.** This is the type
    that needs it, because it is where the two provenances meet: ``onset_logit``,
    ``Vc`` and ``Uc`` descend from ``TensorCues``' grad-bearing tensors and follow the
    encoder onto its accelerator, while ``Jc`` descends from ``contour`` -- numpy by
    ``TensorCues``' own invariant, so CPU-born. The DP adds them together term by term
    (``note_potentials``), so a split here is a hard error the moment the encoder is
    not on CPU. ``Cues``/``TensorCues`` cannot carry this invariant: their tensor-ness
    is about *gradients*, an orthogonal axis, and two of their fields are numpy by
    design. ``__post_init__`` enforces it here rather than letting it surface as a
    stack trace from inside the recurrence."""

    onset_logit: torch.Tensor  # (T,) logit P(onset) per frame
    Vc: torch.Tensor           # (T+1,) cumsum log P(voiced)
    Uc: torch.Tensor           # (T+1,) cumsum log(1 - P(voiced))
    Jc: torch.Tensor           # (T+1,) cumsum of pyin pitch-change indicators
    frame_times: np.ndarray    # (T,) strided grid, absolute seconds
    contour: Contour           # pyin pitch track, for reading note pitches

    def __post_init__(self) -> None:
        _check_one_device(
            "Emissions",
            onset_logit=self.onset_logit, Vc=self.Vc, Uc=self.Uc, Jc=self.Jc,
        )

    @property
    def T(self) -> int:
        return int(self.onset_logit.shape[0])

    @property
    def device(self) -> torch.device:
        """The one device every tensor field lives on (see the class invariant)."""
        return self.onset_logit.device


@dataclass
class BatchEmissions:
    """A minibatch of ``Emissions`` padded to a common length for the batched scan.

    The forward algorithm is a scan (``alpha[j]`` depends on ``alpha[i<j]``) --
    sequential over time like an RNN, but fully parallel across tracks. Padding to a
    common ``Tmax`` turns a per-track Python loop into one ``j``-loop over
    ``(B, ...)`` tensors. ``onset`` pads with 0 and the cumsums pad *flat* (the last
    value repeated), so no voiced/onset mass accrues past a track's true end; the
    scan is read out at each track's own ``lengths[b]``, so the padding never
    contributes to a real partition.

    Carries ``Emissions``' one-device invariant across the batch."""

    onset_logit: torch.Tensor  # (B, Tmax)
    Vc: torch.Tensor           # (B, Tmax+1)
    Uc: torch.Tensor           # (B, Tmax+1)
    Jc: torch.Tensor           # (B, Tmax+1)
    lengths: list[int]

    def __post_init__(self) -> None:
        _check_one_device(
            "BatchEmissions",
            onset_logit=self.onset_logit, Vc=self.Vc, Uc=self.Uc, Jc=self.Jc,
        )

    @property
    def device(self) -> torch.device:
        return self.onset_logit.device


def pad_emissions(emissions: list[Emissions]) -> BatchEmissions:
    """Pad single-track ``Emissions`` into a ``BatchEmissions`` (grad-preserving).

    Each input is internally one-device by construction; this additionally requires
    the *batch* to agree, since the padded buffers are allocated off the first track
    and every other track is scattered into them."""
    _check_one_device(
        "pad_emissions batch",
        **{f"track{b}": e.onset_logit for b, e in enumerate(emissions)},
    )
    lengths = [e.T for e in emissions]
    B, Tmax = len(emissions), max(lengths)
    onset = emissions[0].onset_logit.new_zeros(B, Tmax)
    Vc = onset.new_zeros(B, Tmax + 1)
    Uc = onset.new_zeros(B, Tmax + 1)
    Jc = onset.new_zeros(B, Tmax + 1)
    for b, e in enumerate(emissions):
        t = e.T
        onset[b, :t] = e.onset_logit
        Vc[b, : t + 1], Vc[b, t + 1 :] = e.Vc, e.Vc[-1]  # flat past the end
        Uc[b, : t + 1], Uc[b, t + 1 :] = e.Uc, e.Uc[-1]
        Jc[b, : t + 1], Jc[b, t + 1 :] = e.Jc, e.Jc[-1]
    return BatchEmissions(onset, Vc, Uc, Jc, lengths)


def pitch_jumps(contour: Contour, frame_times: np.ndarray) -> np.ndarray:
    """Per-frame indicator of a pyin pitch-change landing at each grid frame.

    A ``note`` span crossing a change is penalised (``w_pitch``), so the decoder
    splits a legato run at the pitch boundary even with no rest and a weak onset.
    ``group()`` once split on this cue too and no longer does -- pyin's transition
    lags a real onset by ~23 ms, which made it place notes late enough to fall outside
    the match tolerance. Here it is a weighted *feature* rather than a hard boundary,
    so the same lag is something the decoder can learn to discount; that is a weaker
    claim than "this cue is good", and it is why the threshold now lives here. Detection
    bridges pyin dropouts the way ``group()``'s running pitch does: a bass note
    transition usually has a one-frame unvoiced attack right at the boundary, so
    comparing only adjacent voiced frames would miss exactly the changes that matter.
    Each voiced frame is instead compared to the *last* voiced pitch, carried across
    the gap."""
    cmidi = np.full(contour.freqs.shape, np.nan)
    voiced = contour.freqs > 0
    cmidi[voiced] = librosa.hz_to_midi(contour.freqs[voiced])
    nearest = np.clip(
        np.searchsorted(contour.times, frame_times), 0, contour.times.size - 1
    )
    midi = cmidi[nearest]  # pyin pitch on the (strided) CRF grid
    jump = np.zeros(frame_times.size, dtype=np.float32)
    last = np.nan
    for k in range(frame_times.size):
        m = midi[k]
        if np.isnan(m):
            continue  # dropout: hold `last`
        if not np.isnan(last) and abs(m - last) > JUMP_SEMITONES:
            jump[k] = 1.0
        last = m
    return jump


def voiced_logcumsum(
    voicing: np.ndarray | torch.Tensor | None,
    contour: Contour,
    frame_times: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix cumsums of ``(log P(voiced), log P(unvoiced))`` on the strided grid.

    Prefer the voicing head; with none, fall back to pyin's own voiced mask
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability --
    the CRF still runs on DSP-only cues, just without a learned rest signal.
    ``torch.as_tensor`` keeps a numpy voicing detached (inference) but a tensor
    voicing on the graph (training), so grad reaches the encoder's voicing head.
    Both branches land on ``device`` -- the fallback is numpy-born and would otherwise
    pin the cumsums to CPU while a head-fed run put them on the accelerator."""
    if voicing is not None:
        v = _on(voicing, device).float().clamp(1e-6, 1 - 1e-6)
        vlp, ulp = torch.log(v), torch.log1p(-v)
    else:
        mask = resample_mask(contour.times, contour.freqs > 0, frame_times)
        vp = np.clip(mask, 1e-6, 1 - 1e-6)
        vlp = _on(np.log(vp).astype(np.float32), device)
        ulp = _on(np.log1p(-vp).astype(np.float32), device)
    Vc = torch.cat([vlp.new_zeros(1), torch.cumsum(vlp, 0)])
    Uc = torch.cat([ulp.new_zeros(1), torch.cumsum(ulp, 0)])
    return Vc, Uc


def build_emissions(
    cues: Cues | TensorCues, stride: int, device: torch.device | str | None = None
) -> Emissions:
    """Build the strided ``Emissions`` the DP scans from ``Cues`` or ``TensorCues``.

    The two intermediaries share every field name and hold probabilities in the same
    range, so one builder serves both: ``torch.as_tensor`` treats a numpy ``Cues``
    field as a constant (inference/decode) and a tensor ``TensorCues`` field as a
    graph node (training), which is the whole reason the DP is backend-agnostic.

    ``device`` is the single placement every field is built onto, establishing
    ``Emissions``' one-device invariant. Callers that own the potentials pass *their*
    module's device, so the numpy-born ``Jc`` follows the parameters it will be added
    to -- which is what makes a numpy ``Cues`` decodable by a module on an
    accelerator. Defaults to wherever the cues' own evidence already lives."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])
    device = _cues_device(cues) if device is None else torch.device(device)

    # Stride first, place second: the slice is cheap on the source device and only the
    # decimated grid crosses the bus.
    onset = _on(torch.as_tensor(cues.onset).float()[::stride], device)
    onset_logit = torch.logit(onset.clamp(1e-6, 1 - 1e-6))

    voicing = cues.voicing[::stride] if cues.voicing is not None else None
    Vc, Uc = voiced_logcumsum(voicing, cues.contour, frame_times, device)

    # `jump` reads the pyin contour, which is numpy by TensorCues' invariant, so this
    # is the field that would otherwise strand the DP across two devices.
    jump = _on(pitch_jumps(cues.contour, frame_times), device)
    Jc = torch.cat([jump.new_zeros(1), torch.cumsum(jump, 0)])
    return Emissions(onset_logit, Vc, Uc, Jc, frame_times, cues.contour)
