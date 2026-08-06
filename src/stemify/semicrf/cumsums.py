"""Prefix sums over per-frame evidence, so a span score is O(1) in its length.

Every span feature the scorer reads is a sum or mean over ``[i, j)``. Materialising
that per candidate span would be O(dmax) per step; a cumulative sum makes it one
subtraction. These are the cumsums ``Evidence`` carries, built once per batch.

Split out of ``evidence.py``: they change when a *span feature* is added or retired,
which is a different occasion from the ``Evidence`` layout or the encoder's outputs.
"""

from __future__ import annotations

import numpy as np
import torch

from stemify.core.contour import Contour
from stemify.semicrf.tensor_cues import on_device
from stemify.core.grids import resample_mask



def voiced_logcumsum(
    voicing: np.ndarray | torch.Tensor | None,
    contour: Contour | None,
    frame_times: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix cumsums of ``(log P(voiced), log P(unvoiced))`` on the strided grid.

    Prefer the voicing head; with none, fall back to a contour's own voiced mask
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability -- the
    CRF still runs on DSP-only cues, just without a learned rest signal. Both branches
    land on ``device``: the fallback is numpy-born and would otherwise pin the cumsums
    to CPU while a head-fed run put them on the accelerator.

    With *neither*, raise. Voiced/unvoiced is the one thing the segmental scan cannot
    proceed without -- ``rest_mass`` and ``voiced_mass`` are what separate a note from
    a silence -- so defaulting to "everything is voiced" would return a plausible
    number for a model that was never given the question."""
    if voicing is not None:
        v = on_device(voicing, device).float().clamp(1e-6, 1 - 1e-6)
        vlp, ulp = torch.log(v), torch.log1p(-v)
    else:
        if contour is None:
            raise ValueError(
                "no voicing signal: the model has no voicing head and no contour was "
                "supplied, so there is nothing to say where the bass sounds. Train "
                "with a voicing head, or pass a contour to fall back on."
            )
        mask = resample_mask(contour.times, contour.freqs > 0, frame_times)
        vp = np.clip(mask, 1e-6, 1 - 1e-6)
        vlp = on_device(np.log(vp).astype(np.float32), device)
        ulp = on_device(np.log1p(-vp).astype(np.float32), device)
    Vc = torch.cat([vlp.new_zeros(1), torch.cumsum(vlp, 0)])
    Uc = torch.cat([ulp.new_zeros(1), torch.cumsum(ulp, 0)])
    return Vc, Uc


def pitch_cumsums(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-bin pitch logits ``(T, K)`` -> ``(path_c, entropy_c, pitch_c)`` prefix sums.

    Every quantity here has to be additive over frames, because a span reads it as an
    O(1) difference and there are O(T * dmax) spans. That is the constraint, and it is
    why the two features are shaped as they are rather than as some arbitrary function
    of the span's frames.

    ``path`` uses the Bhattacharyya coefficient rather than a dot product: ``BC`` is 1
    for any two *identical* distributions however diffuse, while ``<p, q>`` falls with
    spread, so a dot product would report a quiet passage the head is unsure about as
    constant pitch movement. The first frame has no predecessor and contributes 0."""
    p = torch.softmax(logits, dim=-1)
    # Clamped for the *gradient*, not the value. d(sqrt p)/dp = 1/(2 sqrt p) is
    # unbounded as p -> 0, and a softmax over this many bins underflows to exactly 0
    # in float32 once the head is confident (a logit spread past ~104 does it), so
    # the backward pass hits 1/0 -> inf and then 0 * inf -> NaN while the forward
    # stays perfectly finite. Clamping pins the derivative of the underflowed tail at
    # zero, where it belongs: those bins carry no probability and should not steer
    # anything. Without it the failure is not merely the eventual NaN -- the whole
    # region below ~1e-20 contributes gradients of order 1e10 that swamp the shared
    # trunk, and they grow as the head sharpens, so the pitch branch is punished for
    # becoming confident and is driven back toward uniform.
    sq = p.clamp_min(1e-9).sqrt()
    bc = (sq[1:] * sq[:-1]).sum(-1)                       # (T-1,)
    path = torch.cat([bc.new_zeros(1), 1.0 - bc])         # (T,)
    entropy = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(-1)   # (T,)
    zero_row = p.new_zeros(1, p.shape[-1])
    return (
        torch.cat([path.new_zeros(1), torch.cumsum(path, 0)]),
        torch.cat([entropy.new_zeros(1), torch.cumsum(entropy, 0)]),
        torch.cat([zero_row, torch.cumsum(p, 0)]),
    )
