"""``SemiCRFDecoder``: the ``Decoder`` façade wiring scorer + evidence + DP.

Where ``HeuristicDecoder`` greedily peak-picks the onset curve and thresholds the
voicing curve, the semi-Markov CRF decodes the *whole* track at once: it finds the
single highest-scoring segmentation of the frame grid into ``note`` and ``rest`` spans.
There is no merge pass -- a learned duration potential makes fragmentation globally
costly, so the over-segmentation the heuristic repairs after the fact is instead priced
into the objective and never produced.

Two faces over the same recurrence:

  - ``decode(cues) -> notes`` runs segmental **Viterbi** for the single best
    segmentation, then reads a pitch per ``note`` span. The ``Decoder`` interface.
  - ``nll(...)`` / ``nll_batch(...)`` / ``nll_chunk(...)`` run the segmental **forward**
    algorithm for ``log Z`` and subtract the gold score. With a ``TensorCues`` the
    gradient reaches the encoder; with a numpy ``Cues`` it is an inference-only score.

Pitch is never invented: it is read from the contour after decoding, exactly as the
heuristic does. See ``docs/semi-crf.md``, which carries the theory and a ledger of
where this implementation departs from it.
"""

import librosa
import numpy as np
import torch
from torch import nn

from stemify.cues import Cues
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, NOTE, STRIDE
from stemify.decoders.semicrf.evidence import Evidence, TensorCues, build_evidence
from stemify.decoders.semicrf.forward import chunk_log_partition, log_partitions
from stemify.decoders.semicrf.gold import gold_segments
from stemify.decoders.semicrf.score import segmentation_scores
from stemify.decoders.semicrf.spans import LinearSpanScorer, SpanScorer
from stemify.decoders.semicrf.viterbi import viterbi
from stemify.notes import NoteEvent


class SemiCRFDecoder(Decoder, nn.Module):
    """A trainable semi-CRF decoder: segmental Viterbi to ``decode``, segmental forward
    for the ``nll`` that trains it end to end with the encoder.

    ``scorer`` is injected rather than constructed inline, because *what a span is
    worth* is the axis this model is expected to change along -- a neural potential is
    a different ``SpanScorer``, not a different decoder.
    """

    name = "semicrf"

    def __init__(
        self,
        *,
        stride: int = STRIDE,
        dmax_s: float = DMAX_S,
        dur_edges_s: np.ndarray = DUR_EDGES_S,
        scorer: SpanScorer | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.stride = stride
        self.scorer = scorer or LinearSpanScorer(dmax_s, dur_edges_s)

    @property
    def device(self) -> torch.device:
        """Where the scorer lives -- and so where the evidence is built.

        The module is the authority on placement, not the cues: a numpy ``Cues`` has no
        device of its own, and the DP adds its evidence straight to the parameters.
        Sourcing it here is what lets a CUDA-resident decoder score CPU-born cues."""
        return next(self.scorer.parameters()).device

    def _prepare(self, cues_list: list[Cues | TensorCues]) -> tuple[Evidence, object]:
        """Evidence for a batch, and the ``Grid`` its tracks share.

        All tracks come off one feature hop, so the grid is derived once from the first
        rather than per track."""
        ev = build_evidence(cues_list, self.stride, self.device)
        return ev, self.scorer.grid(ev.frame_times[0])

    def nll(self, cues: Cues | TensorCues, notes: list[NoteEvent]) -> torch.Tensor:
        """``log Z - score(gold)`` for one track -- the batch of one."""
        return self.nll_batch([cues], [notes])[0]

    def nll_batch(
        self,
        cues_list: list[Cues | TensorCues],
        notes_list: list[list[NoteEvent]],
    ) -> torch.Tensor:
        """Per-track NLL ``(B,)``: one batched ``log Z`` scan over the padded evidence,
        minus each track's gold score.

        Both halves read the *same* padded batch: gold indices never reach past a
        track's true length, so the flat padding beyond it is never gathered."""
        ev, grid = self._prepare(cues_list)
        golds = [
            gold_segments(ft, n, grid.dmax)
            for ft, n in zip(ev.frame_times, notes_list, strict=True)
        ]
        return log_partitions(self.scorer, ev, grid) - segmentation_scores(
            self.scorer, ev, golds, grid
        )

    def nll_chunk(
        self,
        cues_list: list[Cues | TensorCues],
        golds: list[list[tuple[int, int, int]]],
        carry: torch.Tensor | None,
        context: int,
        right: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One truncated-BPTT step: ``(A, gold_score, window)`` for a batch of chunks.

        Deliberately *not* an NLL: ``A`` is the cumulative log-mass over everything
        scanned so far, so the caller forms the per-chunk loss by telescoping,
        ``(A_k - A_{k-1}) - gold_score_k``.

        ``golds`` are *window-local* segmentations supplied by the dataset rather than
        derived from notes here: gold is built once over the whole track and split by
        chunk, which is what makes the pieces tile it exactly and the losses telescope.
        Deriving it per chunk would re-run the audible-gap rule on a truncated view and
        silently disagree at the seams."""
        ev, grid = self._prepare(cues_list)
        A, window = chunk_log_partition(self.scorer, ev, grid, carry, context, right)
        return A, segmentation_scores(self.scorer, ev, golds, grid), window

    def decode(self, cues: Cues) -> list[NoteEvent]:
        """The notes implied by ``cues`` under the learned scorer (Viterbi)."""
        ev, grid = self._prepare([cues])
        if ev.lengths[0] == 0:
            return []
        return self._segs_to_notes(viterbi(self.scorer, ev, grid), ev)

    def _segs_to_notes(
        self, segs: list[tuple[int, int, int]], ev: Evidence
    ) -> list[NoteEvent]:
        """Turn ``note`` segments into ``NoteEvent``s, reading each note's pitch as the
        median contour pitch over its span -- pitch comes from the contour, never
        invented. A span the contour left entirely pitchless yields no note."""
        ft, contour = ev.frame_times[0], ev.contours[0]
        hop = float(np.median(np.diff(ft))) if ft.size > 1 else 0.01 * self.stride
        ct, cf = contour.times, contour.freqs
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
