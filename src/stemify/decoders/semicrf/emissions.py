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
from stemify.decoders.heuristic import PITCH_SPLIT_SEMITONES
from stemify.grids import resample_mask


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
    what keeps the scan linear in the segment cap."""

    onset_logit: torch.Tensor  # (T,) logit P(onset) per frame
    Vc: torch.Tensor           # (T+1,) cumsum log P(voiced)
    Uc: torch.Tensor           # (T+1,) cumsum log(1 - P(voiced))
    Jc: torch.Tensor           # (T+1,) cumsum of pyin pitch-change indicators
    frame_times: np.ndarray    # (T,) strided grid, absolute seconds
    contour: Contour           # pyin pitch track, for reading note pitches

    @property
    def T(self) -> int:
        return int(self.onset_logit.shape[0])


@dataclass
class BatchEmissions:
    """A minibatch of ``Emissions`` padded to a common length for the batched scan.

    The forward algorithm is a scan (``alpha[j]`` depends on ``alpha[i<j]``) --
    sequential over time like an RNN, but fully parallel across tracks. Padding to a
    common ``Tmax`` turns a per-track Python loop into one ``j``-loop over
    ``(B, ...)`` tensors. ``onset`` pads with 0 and the cumsums pad *flat* (the last
    value repeated), so no voiced/onset mass accrues past a track's true end; the
    scan is read out at each track's own ``lengths[b]``, so the padding never
    contributes to a real partition."""

    onset_logit: torch.Tensor  # (B, Tmax)
    Vc: torch.Tensor           # (B, Tmax+1)
    Uc: torch.Tensor           # (B, Tmax+1)
    Jc: torch.Tensor           # (B, Tmax+1)
    lengths: list[int]


def pad_emissions(emissions: list[Emissions]) -> BatchEmissions:
    """Pad single-track ``Emissions`` into a ``BatchEmissions`` (grad-preserving)."""
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
    splits a legato run at the pitch boundary even with no rest and a weak onset --
    the third split cue ``group()`` uses, restored to the global decoder. Detection
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
        if not np.isnan(last) and abs(m - last) > PITCH_SPLIT_SEMITONES:
            jump[k] = 1.0
        last = m
    return jump


def voiced_logcumsum(
    voicing: np.ndarray | torch.Tensor | None,
    contour: Contour,
    frame_times: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix cumsums of ``(log P(voiced), log P(unvoiced))`` on the strided grid.

    Prefer the voicing head; with none, fall back to pyin's own voiced mask
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability --
    the CRF still runs on DSP-only cues, just without a learned rest signal.
    ``torch.as_tensor`` keeps a numpy voicing detached (inference) but a tensor
    voicing on the graph (training), so grad reaches the encoder's voicing head."""
    if voicing is not None:
        v = torch.as_tensor(voicing).float().clamp(1e-6, 1 - 1e-6)
        vlp, ulp = torch.log(v), torch.log1p(-v)
    else:
        mask = resample_mask(contour.times, contour.freqs > 0, frame_times)
        vp = np.clip(mask, 1e-6, 1 - 1e-6)
        vlp = torch.from_numpy(np.log(vp).astype(np.float32))
        ulp = torch.from_numpy(np.log1p(-vp).astype(np.float32))
    Vc = torch.cat([vlp.new_zeros(1), torch.cumsum(vlp, 0)])
    Uc = torch.cat([ulp.new_zeros(1), torch.cumsum(ulp, 0)])
    return Vc, Uc


def build_emissions(cues: Cues | TensorCues, stride: int) -> Emissions:
    """Build the strided ``Emissions`` the DP scans from ``Cues`` or ``TensorCues``.

    The two intermediaries share every field name and hold probabilities in the same
    range, so one builder serves both: ``torch.as_tensor`` treats a numpy ``Cues``
    field as a constant (inference/decode) and a tensor ``TensorCues`` field as a
    graph node (training), which is the whole reason the DP is backend-agnostic."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])

    onset = torch.as_tensor(cues.onset).float()[::stride]
    onset_logit = torch.logit(onset.clamp(1e-6, 1 - 1e-6))

    voicing = cues.voicing[::stride] if cues.voicing is not None else None
    Vc, Uc = voiced_logcumsum(voicing, cues.contour, frame_times)

    jump = pitch_jumps(cues.contour, frame_times)
    Jc = torch.cat([torch.zeros(1), torch.cumsum(torch.from_numpy(jump), 0)])
    return Emissions(onset_logit, Vc, Uc, Jc, frame_times, cues.contour)
