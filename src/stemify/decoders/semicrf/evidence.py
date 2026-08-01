"""``Cues | TensorCues`` -> the per-frame tensors the DP scans, and how spans read them.

``Evidence`` is **always batched**: one track is a batch of one. There is no separate
single-track type, because the two differed only in a leading axis while duplicating
every accessor and forcing the alpha window to be shape-polymorphic.

Voicing and the pitch-change cue enter as *prefix cumsums*, so a span's mass is an
O(1) difference ``C[j] - C[i]`` rather than a re-sum per candidate. That is what keeps
the scan linear in the segment cap, and it is the constraint every future feature has
to satisfy -- an interior term must be additive over frames, or endpoint- and
length-only.

The span accessors live here rather than in the scorer because they encode *index
conventions*, not modelling choices. In particular ``interior_jumps`` owns the
``i + 1`` offset that makes a pitch change at the span's own first frame not count
against it; that offset used to be applied by three different callers, one of which
delegated it to *its* caller in a docstring.

``torch.as_tensor`` is the load-bearing trick throughout: it treats a numpy ``Cues``
field as a constant (inference/decode) and a ``TensorCues`` tensor as a graph node
(training), so one builder serves both and the DP is backend-agnostic.
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
# ``heuristic.PITCH_SPLIT_SEMITONES`` until that constant was turned off by default (a
# pitch change no longer *places* a note there -- see its docstring). The two usages
# were never the same question: there the threshold decided a hard note boundary, here
# it only raises a feature the CRF is free to weight, so the value is pinned locally at
# what every recorded semi-CRF run used rather than following a default that now means
# something else.
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

    Field-for-field a ``Cues``, but ``onset``/``voicing`` are grad-bearing tensors (the
    encoder's forward output) rather than detached numpy, so a loss over them
    backprops into the encoder. ``contour`` and ``frame_times`` stay numpy: they are
    constants (pyin's pitch, the time grid) that never need gradients. ``Cues`` remains
    the *inference* intermediary; this is only built inside a training step. It lives
    here, not beside ``Cues``, so ``cues.py`` stays torch-free and the light DSP path
    never pulls torch."""

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
class Evidence:
    """The strided per-frame evidence for a batch of tracks, as the DP consumes it.

    **Device invariant: all four tensor fields live on one device.** This is the type
    that needs it, because it is where two provenances meet: ``onset_logit``,
    ``voiced_c`` and ``rest_c`` descend from ``TensorCues``' grad-bearing tensors and
    follow the encoder onto its accelerator, while ``jump_c`` descends from a numpy
    contour and is CPU-born. The DP adds them term by term, so a split is a hard error
    the moment the encoder is not on CPU.

    Ragged tracks are padded to a common length: ``onset_logit`` pads with 0 and the
    cumsums pad **flat** (last value repeated), so no voiced or onset mass accrues past
    a track's true end. The scan reads each track out at its own ``lengths[b]``.
    """

    onset_logit: torch.Tensor       # (B, Tmax)
    voiced_c: torch.Tensor          # (B, Tmax+1) cumsum log P(voiced)
    rest_c: torch.Tensor            # (B, Tmax+1) cumsum log(1 - P(voiced))
    jump_c: torch.Tensor            # (B, Tmax+1) cumsum of pitch-change indicators
    lengths: list[int]
    frame_times: list[np.ndarray]   # per track, strided grid, absolute seconds
    contours: list[Contour]         # per track, for reading note pitches

    def __post_init__(self) -> None:
        _check_one_device(
            "Evidence",
            onset_logit=self.onset_logit, voiced_c=self.voiced_c,
            rest_c=self.rest_c, jump_c=self.jump_c,
        )

    @property
    def B(self) -> int:
        return len(self.lengths)

    @property
    def T(self) -> int:
        """The padded width. Use ``lengths[b]`` for a track's own extent."""
        return int(self.onset_logit.shape[1])

    @property
    def device(self) -> torch.device:
        return self.onset_logit.device

    def one(self, b: int = 0) -> "Evidence":
        """Row ``b`` as its own batch of one -- for the single-track paths (Viterbi,
        decode) that would otherwise index every field by hand."""
        t = self.lengths[b]
        return Evidence(
            self.onset_logit[b : b + 1, :t], self.voiced_c[b : b + 1, : t + 1],
            self.rest_c[b : b + 1, : t + 1], self.jump_c[b : b + 1, : t + 1],
            [t], [self.frame_times[b]], [self.contours[b]],
        )

    # --- span accessors: the index conventions, in one place --------------------

    def onset_at(self, b, i) -> torch.Tensor:
        """Onset logit at a span's first frame."""
        return self.onset_logit[b, i]

    def voiced_mass(self, b, i, j) -> torch.Tensor:
        """``sum of log P(voiced)`` over ``[i, j)``."""
        return self.voiced_c[b, j] - self.voiced_c[b, i]

    def rest_mass(self, b, i, j) -> torch.Tensor:
        """``sum of log(1 - P(voiced))`` over ``[i, j)``."""
        return self.rest_c[b, j] - self.rest_c[b, i]

    def interior_jumps(self, b, i, j) -> torch.Tensor:
        """Pitch changes landing **strictly inside** ``[i, j)``.

        The ``i + 1`` is the whole point: a pitch change at the span's own first frame
        is what *starts* this note, so charging the span for it would penalise every
        correctly-placed boundary."""
        return self.jump_c[b, j] - self.jump_c[b, i + 1]


def pitch_jumps(contour: Contour, frame_times: np.ndarray) -> np.ndarray:
    """Per-frame indicator of a pyin pitch-change landing at each grid frame.

    A ``note`` span crossing a change is penalised, so the decoder can split a legato
    run at the pitch boundary even with no rest and a weak onset. ``group()`` once
    split on this cue too and no longer does -- pyin's transition lags a real onset,
    which made it place notes late enough to fall outside the match tolerance. Here it
    is a weighted *feature* rather than a hard boundary, so the same lag is something
    the decoder can learn to discount; that is a weaker claim than "this cue is good",
    and it is why the threshold lives here rather than being imported.

    Detection bridges pyin dropouts the way ``group()``'s running pitch does: a bass
    note transition usually has a one-frame unvoiced attack right at the boundary, so
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
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability -- the
    CRF still runs on DSP-only cues, just without a learned rest signal. Both branches
    land on ``device``: the fallback is numpy-born and would otherwise pin the cumsums
    to CPU while a head-fed run put them on the accelerator."""
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


def _track_tensors(
    cues: Cues | TensorCues, stride: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """One track's ``(onset_logit, voiced_c, rest_c, jump_c, frame_times)``."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])
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
    return onset_logit, Vc, Uc, Jc, frame_times


def build_evidence(
    cues: Cues | TensorCues | list[Cues | TensorCues],
    stride: int,
    device: torch.device | str | None = None,
) -> Evidence:
    """Build the strided ``Evidence`` the DP scans, from one track or a list.

    ``device`` is the single placement every field is built onto, establishing the
    one-device invariant. Callers that own the scorer pass *its* module's device, so
    the numpy-born ``jump_c`` follows the parameters it will be added to -- which is
    what makes a numpy ``Cues`` decodable by a module on an accelerator.

    Padding is grad-preserving: the per-track tensors are scattered into buffers
    allocated off the first track."""
    batch = cues if isinstance(cues, list) else [cues]
    if not batch:
        raise ValueError("build_evidence needs at least one track")
    device = _cues_device(batch[0]) if device is None else torch.device(device)

    built = [_track_tensors(c, stride, device) for c in batch]
    lengths = [int(t[0].shape[0]) for t in built]
    B, Tmax = len(built), max(lengths)

    onset = built[0][0].new_zeros(B, Tmax)
    Vc = onset.new_zeros(B, Tmax + 1)
    Uc = onset.new_zeros(B, Tmax + 1)
    Jc = onset.new_zeros(B, Tmax + 1)
    for b, (o, v, u, jc, _) in enumerate(built):
        t = lengths[b]
        onset[b, :t] = o
        Vc[b, : t + 1], Vc[b, t + 1 :] = v, v[-1]  # flat past the end
        Uc[b, : t + 1], Uc[b, t + 1 :] = u, u[-1]
        Jc[b, : t + 1], Jc[b, t + 1 :] = jc, jc[-1]
    return Evidence(
        onset, Vc, Uc, Jc, lengths,
        [t[4] for t in built], [c.contour for c in batch],
    )
