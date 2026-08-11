"""Prefix sums over per-frame evidence, so a span score is O(1) in its length.

Every span feature the scorer reads is a sum or mean over ``[start, end)``. Materialising
that per candidate span would be O(dmax) per step; a cumulative sum makes it one
subtraction. These are the cumsums ``Evidence`` carries, built once per batch.

Split out of ``evidence.py``: they change when a *span feature* is added or retired,
which is a different occasion from the ``Evidence`` layout or the encoder's outputs.
"""

from __future__ import annotations

import numpy as np
import torch

from stemify.music.contour import Contour
from stemify.audio.features import BINS_PER_SEMITONE
from stemify.semicrf.placement import on_device
from stemify.music.grids import resample_mask



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
    proceed without -- it is what separates a note from a silence -- so defaulting to
    "everything is voiced" would return a plausible number for a model that was never
    given the question."""
    if voicing is not None:
        voiced = on_device(voicing, device).float().clamp(1e-6, 1 - 1e-6)
        voiced_lp, rest_lp = torch.log(voiced), torch.log1p(-voiced)
    else:
        if contour is None:
            raise ValueError(
                "no voicing signal: the model has no voicing head and no contour was "
                "supplied, so there is nothing to say where the bass sounds. Train "
                "with a voicing head, or pass a contour to fall back on."
            )
        mask = resample_mask(contour.times, contour.freqs > 0, frame_times)
        voiced = np.clip(mask, 1e-6, 1 - 1e-6)
        voiced_lp = on_device(np.log(voiced).astype(np.float32), device)
        rest_lp = on_device(np.log1p(-voiced).astype(np.float32), device)
    voiced_c = torch.cat([voiced_lp.new_zeros(1), torch.cumsum(voiced_lp, 0)])
    rest_c = torch.cat([rest_lp.new_zeros(1), torch.cumsum(rest_lp, 0)])
    return voiced_c, rest_c


def pitch_logprob_cumsum(
    logits: torch.Tensor, n_pitches: int, bins_per_semitone: int = BINS_PER_SEMITONE
) -> torch.Tensor:
    """Per-bin pitch logits ``(T, K)`` -> ``(T+1, n_pitches)`` cumsum of semitone log-probs.

    **The potential that lets the CRF supervise pitch value at all.** Under a
    ``{note, rest}`` alphabet nothing in the structured objective says which pitch is
    right -- the label names none. A span labelled ``note@p`` can read the head's own
    log-probability of ``p`` across the span, and then ``d NLL / d(pitch head)`` is a
    segment-level pitch signal instead of nothing.

    Bins are pooled to semitones by ``logsumexp``, which is the marginal "the pitch lies
    somewhere in this semitone" rather than a pick of the centre bin. That matters
    because ``two_hot`` deliberately splits a note's mass across the two bins straddling
    its true fractional pitch, so the centre bin alone under-counts every note not
    exactly at a semitone -- systematically, and worst for the in-tune ones.

    The axis is truncated to a whole number of semitones *before* the softmax, so the
    result is a proper distribution over the pitches the alphabet can name rather than
    one leaking mass into bins no label reads.

    Cumulative because a span reads it as ``(C[end] - C[start]) / (end - start)`` --
    the **mean**
    log-probability, so a geometric mean of the frame posteriors. That is the right
    aggregate here and the arithmetic mean is not: one frame confidently disagreeing
    should veto a span's pitch, which is what a geometric mean does and a mixture
    deliberately does not.
    """
    need = n_pitches * bins_per_semitone
    if logits.shape[-1] + 1 < need:
        raise ValueError(
            f"pitch head emits {logits.shape[-1]} bins, which cannot cover the "
            f"{n_pitches} semitones the alphabet names at {bins_per_semitone} "
            "bins/semitone. The label set is derived from the head's axis "
            "(LabelSet.from_bins); they have drifted."
        )
    logprob = torch.log_softmax(logits, dim=-1)
    # **Pooling blocks are centred on the semitone, which costs one bin of padding.**
    # Semitone `s` sits at bin `bins_per_semitone * s` exactly (that is what
    # `LabelSet.bin_of` returns), so its block has to be the bins *around* that, not
    # the ones above it. Reshaping the raw axis would give `[3s, 3s+3)` -- half a
    # semitone sharp -- and a note tuned slightly flat would be credited to the
    # semitone below. Padding one `log 0` bin at the bottom shifts every block down by
    # one, making them `[3s-1, 3s+2)`.
    pad = logprob.new_full((logprob.shape[0], 1), float("-inf"))
    blocks = torch.cat([pad, logprob], dim=-1)[:, :need]
    pooled = torch.logsumexp(
        blocks.reshape(logprob.shape[0], n_pitches, bins_per_semitone), dim=-1
    )                                                         # (T, n_pitches)
    return torch.cat([pooled.new_zeros(1, n_pitches), torch.cumsum(pooled, 0)])
