"""``SemiCRFDecoder``: the ``Decoder`` façade wiring potentials + emissions + DP.

Where ``HeuristicDecoder`` greedily peak-picks the onset curve and thresholds the
voicing curve, the semi-Markov CRF decodes the *whole* track at once: it finds the
single highest-scoring segmentation of the frame grid into ``note`` and ``rest``
spans under a handful of learned potentials. There is no merge pass -- a learned
duration potential makes fragmentation globally costly, so the over-segmentation the
heuristic repairs after the fact is instead priced into the objective and never
produced.

Two faces over the same recurrence:

  - ``decode(cues) -> notes`` runs segmental **Viterbi** for the single best
    segmentation, then reads a pitch off ``cues.contour`` per ``note`` span. The
    ``Decoder`` interface.
  - ``nll(cues, notes)`` / ``nll_batch(...)`` run the segmental **forward** algorithm
    for ``log Z`` and subtract the gold score; ``log Z - gold`` is the NLL, fully
    differentiable. In training the cues are a ``TensorCues`` (grad-bearing tensors
    from the encoder's heads), so the gradient reaches the encoder; a numpy ``Cues``
    gives an inference-only score.

Pitch is never invented: it is read from ``cues.contour`` after decoding, exactly as
the heuristic does. See ``docs/semi-crf.md`` and the sibling modules for the pieces
this composes: ``potentials``, ``emissions``, ``window``, ``forward``, ``viterbi``,
``gold`` (which segmentation is gold) and ``score`` (what a segmentation is worth).
"""

import librosa
import numpy as np
import torch
from torch import nn

from stemify.cues import Cues
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, STRIDE
from stemify.decoders.semicrf.emissions import (
    Emissions,
    TensorCues,
    build_emissions,
    pad_emissions,
)
from stemify.decoders.semicrf.forward import (
    batched_log_partition,
    chunk_log_partition,
    log_partition,
)
from stemify.decoders.semicrf.gold import gold_segments
from stemify.decoders.semicrf.potentials import SemiCRFPotentials
from stemify.decoders.semicrf.score import segmentation_score, segmentation_scores
from stemify.decoders.semicrf.viterbi import viterbi
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

    @property
    def device(self) -> torch.device:
        """Where the potentials live -- and so where the emissions are built.

        The module is the authority on placement, not the cues: a numpy ``Cues`` has
        no device of its own, and the DP adds its evidence straight to the parameters.
        Sourcing it here is what lets a CUDA-resident decoder score CPU-born cues, and
        keeps ``Emissions``' one-device invariant aligned with the parameters rather
        than merely self-consistent."""
        return self.pot.dur.device

    def nll(self, cues: Cues | TensorCues, notes: list[NoteEvent]) -> torch.Tensor:
        """Negative log-likelihood of the gold segmentation (from ``notes``) under
        the model, given ``cues``: ``log Z - score(gold)``. The differentiable
        training objective -- with a ``TensorCues`` (the training intermediary),
        backprop reaches the encoder that produced it; with a numpy ``Cues`` it is
        differentiable only in the CRF's own potentials (an eval/inference score)."""
        e = build_emissions(cues, self.stride, self.device)
        dmax, dur_bucket = self.pot.grid(e.frame_times)
        gold = gold_segments(e.frame_times, notes, dmax)
        return log_partition(self.pot, e, dmax, dur_bucket) - segmentation_score(
            self.pot, e, gold, dur_bucket
        )

    def nll_batch(
        self,
        cues_list: list[Cues | TensorCues],
        notes_list: list[list[NoteEvent]],
    ) -> torch.Tensor:
        """Per-track NLL ``(B,)`` for a minibatch: one batched ``log Z`` scan over the
        padded emissions, minus each track's gold score. ``SemiCRFLoss`` means these;
        grad reaches the encoder through each ``TensorCues``. All tracks share one
        feature hop, so ``dmax``/``dur_bucket`` are computed once.

        Both halves read the *same* padded batch: gold indices never reach past a
        track's true length, so the flat padding beyond it is never gathered."""
        emissions = [build_emissions(c, self.stride, self.device) for c in cues_list]
        dmax, dur_bucket = self.pot.grid(emissions[0].frame_times)
        batch = pad_emissions(emissions)
        logZ = batched_log_partition(self.pot, batch, dmax, dur_bucket)
        golds = [
            gold_segments(e.frame_times, n, dmax)
            for e, n in zip(emissions, notes_list, strict=True)
        ]
        return logZ - segmentation_scores(self.pot, batch, golds, dur_bucket)

    def nll_chunk(
        self,
        cues_list: list[Cues | TensorCues],
        golds: list[list[tuple[int, int, int]]],
        carry: torch.Tensor | None,
        context: int,
        right: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One truncated-BPTT step: ``(A, gold_score, window)`` for a batch of chunks.

        The chunked analogue of ``nll_batch``, and deliberately *not* an NLL: ``A`` is
        the cumulative log-mass over everything scanned so far, so the caller forms the
        per-chunk loss by telescoping, ``(A_k - A_{k-1}) - gold_score_k``. Summed over
        a track's chunks that is exactly ``log Z - score(gold)``, while each term's
        gradient reaches only its own chunk's activations.

        ``golds`` are *window-local* segmentations supplied by the dataset rather than
        derived from notes here: gold is built once over the whole track and split by
        chunk, which is what makes the pieces tile it exactly and the losses telescope.
        Deriving it per chunk would re-run the audible-gap rule on a truncated view and
        silently disagree at the seams.

        ``carry`` is the previous chunk's window (``None`` at a track's start), and
        ``context``/``right`` mark the frames present only as margin -- see
        ``forward.chunk_log_partition``.
        """
        emissions = [build_emissions(c, self.stride, self.device) for c in cues_list]
        dmax, dur_bucket = self.pot.grid(emissions[0].frame_times)
        batch = pad_emissions(emissions)
        A, window = chunk_log_partition(
            self.pot, batch, dmax, dur_bucket, carry, context, right
        )
        return A, segmentation_scores(self.pot, batch, golds, dur_bucket), window

    def decode(self, cues: Cues) -> list[NoteEvent]:
        """The notes implied by ``cues`` under the learned potentials (Viterbi)."""
        e = build_emissions(cues, self.stride, self.device)
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
