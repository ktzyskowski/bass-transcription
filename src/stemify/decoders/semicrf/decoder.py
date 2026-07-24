"""``SemiCRFDecoder``: the ``Decoder`` façade wiring potentials + emissions + DP.

Where ``HeuristicDecoder`` greedily peak-picks the onset curve and thresholds the
voicing curve, the semi-Markov CRF decodes the *whole* track at once: it finds the
single highest-scoring segmentation of the frame grid into ``note`` and ``rest``
spans under a handful of learned potentials. There is no merge pass -- a learned
duration potential makes fragmentation globally costly, so the over-segmentation the
heuristic repairs after the fact is instead priced into the objective and never
produced.

Two faces, one recurrence (``dp``):

  - ``decode(cues) -> notes`` runs segmental **Viterbi** for the single best
    segmentation, then reads a pitch off ``cues.contour`` per ``note`` span. The
    ``Decoder`` interface.
  - ``nll(cues, notes)`` / ``nll_batch(...)`` run the segmental **forward** algorithm
    for ``log Z`` and subtract the gold score; ``log Z - gold`` is the NLL, fully
    differentiable. In training the cues are a ``TensorCues`` (grad-bearing tensors
    from the encoder's heads), so the gradient reaches the encoder; a numpy ``Cues``
    gives an inference-only score.

Pitch is never invented: it is read from ``cues.contour`` after decoding, exactly as
the heuristic does. See ``docs/semi-crf.md`` and the sibling modules (``potentials``,
``emissions``, ``gold``, ``dp``) for the pieces this composes.
"""

import librosa
import numpy as np
import torch
from torch import nn

from stemify.cues import Cues
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, STRIDE
from stemify.decoders.semicrf.dp import (
    batched_log_partition,
    gold_score,
    log_partition,
    viterbi,
)
from stemify.decoders.semicrf.emissions import (
    Emissions,
    TensorCues,
    build_emissions,
    pad_emissions,
)
from stemify.decoders.semicrf.gold import gold_segments
from stemify.decoders.semicrf.potentials import SemiCRFPotentials
from stemify.notes import NoteEvent


class SemiCRFDecoder(Decoder, nn.Module):
    """A trainable semi-CRF decoder: segmental Viterbi to ``decode``, segmental
    forward for the ``nll`` that trains it end to end with the encoder."""

    name = "semicrf"

    def __init__(
        self,
        *,
        stride: int = STRIDE,
        dmax_s: float = DMAX_S,
        dur_edges_s: np.ndarray = DUR_EDGES_S,
    ) -> None:
        nn.Module.__init__(self)
        self.stride = stride
        self.pot = SemiCRFPotentials(dmax_s, dur_edges_s)

    def nll(self, cues: Cues | TensorCues, notes: list[NoteEvent]) -> torch.Tensor:
        """Negative log-likelihood of the gold segmentation (from ``notes``) under
        the model, given ``cues``: ``log Z - score(gold)``. The differentiable
        training objective -- with a ``TensorCues`` (the training intermediary),
        backprop reaches the encoder that produced it; with a numpy ``Cues`` it is
        differentiable only in the CRF's own potentials (an eval/inference score)."""
        e = build_emissions(cues, self.stride)
        dmax, dur_bucket = self.pot.grid(e.frame_times)
        gold = gold_segments(e.frame_times, notes, dmax)
        return log_partition(self.pot, e, dmax, dur_bucket) - gold_score(
            self.pot, e, gold, dur_bucket
        )

    def nll_batch(
        self,
        cues_list: list[Cues | TensorCues],
        notes_list: list[list[NoteEvent]],
    ) -> torch.Tensor:
        """Per-track NLL ``(B,)`` for a minibatch: one batched ``log Z`` scan over the
        padded emissions, minus each track's gold score (gold is cheap and per-track).
        ``SemiCRFLoss`` means these; grad reaches the encoder through each
        ``TensorCues``. All tracks share one feature hop, so ``dmax``/``dur_bucket``
        are computed once."""
        emissions = [build_emissions(c, self.stride) for c in cues_list]
        dmax, dur_bucket = self.pot.grid(emissions[0].frame_times)
        logZ = batched_log_partition(self.pot, pad_emissions(emissions), dmax, dur_bucket)
        gold = torch.stack([
            gold_score(self.pot, e, gold_segments(e.frame_times, n, dmax), dur_bucket)
            for e, n in zip(emissions, notes_list)
        ])
        return logZ - gold

    def decode(self, cues: Cues) -> list[NoteEvent]:
        """The notes implied by ``cues`` under the learned potentials (Viterbi)."""
        e = build_emissions(cues, self.stride)
        if e.T == 0:
            return []
        dmax, dur_bucket = self.pot.grid(e.frame_times)
        return self._segs_to_notes(viterbi(self.pot, e, dmax, dur_bucket), e)

    def _segs_to_notes(
        self, segs: list[tuple[int, int, int]], e: Emissions
    ) -> list[NoteEvent]:
        """Turn ``note`` segments into ``NoteEvent``s, reading each note's pitch as
        the median pyin pitch over its span -- pitch comes from the contour, never
        invented. A span pyin left entirely pitchless yields no note."""
        ft = e.frame_times
        hop = float(np.median(np.diff(ft))) if ft.size > 1 else 0.01 * self.stride
        ct, cf = e.contour.times, e.contour.freqs
        cmidi = np.full(cf.shape, np.nan)
        voiced = cf > 0
        cmidi[voiced] = librosa.hz_to_midi(cf[voiced])
        notes: list[NoteEvent] = []
        for label, i, j in segs:
            if label != NOTE:
                continue
            onset, offset = float(ft[i]), float(ft[j - 1] + hop)
            sel = (ct >= onset) & (ct < offset)
            if not sel.any() or np.isnan(cmidi[sel]).all():
                continue
            pitch = float(np.nanmedian(cmidi[sel]))
            if np.isnan(pitch):
                continue
            notes.append(NoteEvent(onset=onset, offset=offset, midi_pitch=pitch))
        return notes
