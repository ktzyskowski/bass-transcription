"""``SemiCRFDecoder``: the ``Decoder`` façade wiring scorer + evidence + DP.

Where ``GreedyDecoder`` greedily peak-picks the onset curve and thresholds the
voicing curve, the semi-Markov CRF decodes the *whole* track at once: it finds the
single highest-scoring segmentation of the frame grid into ``note`` and ``rest`` spans.
There is no merge pass -- a learned duration potential makes fragmentation globally
costly, so the over-segmentation the greedy decoder repairs after the fact is instead priced
into the objective and never produced.

Two faces over the same recurrence:

  - ``decode(frames) -> notes`` runs segmental **Viterbi** for the single best
    segmentation, then reads a pitch per ``note`` span. The ``Decoder`` interface.
  - ``nll(...)`` / ``nll_batch(...)`` / ``nll_chunk(...)`` run the segmental **forward**
    algorithm for ``log Z`` and subtract the gold score. Over grad-bearing ``Frames``
    the gradient reaches the encoder; over numpy ones it is an inference-only score.

Pitch is never invented: a pitch alphabet names it in the label the DP scored, and a
{note, rest} alphabet reads it off the contour, as the greedy decoder does. See
``docs/semi-crf.md`` for the theory and a ledger of where this departs from it.
"""

from collections.abc import Callable, Mapping, Sequence

import librosa
import numpy as np
import torch
from torch import nn

from stemify.transcription.frames import Frames
from stemify.transcription.decoder import Decoder
from stemify.semicrf.constants import DMAX_S, DUR_EDGES_S, STRIDE
from stemify.semicrf.evidence import Evidence, build_evidence
from stemify.semicrf.forward import chunk_log_partition, log_partitions
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.score import segmentation_scores
from stemify.semicrf.spans import Grid, LinearSpanScorer, SpanScorer
from stemify.semicrf.marginals import segment_confidence, span_marginals
from stemify.semicrf.viterbi import viterbi
from stemify.music.notes import NoteEvent


class SemiCRFDecoder(Decoder, nn.Module):
    """A trainable semi-CRF decoder: segmental Viterbi to ``decode``, segmental forward
    for the ``nll`` that trains it end to end with the encoder.

    ``scorer`` is injected rather than constructed inline, because *what a span is
    worth* is the axis this model is expected to change along -- a neural potential is
    a different ``SpanScorer``, not a different decoder.
    """

    name = "semicrf"

    # Declared, not assigned: a submodule assigned in ``__init__`` is otherwise read
    # back through ``nn.Module.__getattr__``, whose return type is ``Tensor | Module``.
    # Callers reading the scorer's *geometry* (``dmax_s``, ``dur_edges``) then get a
    # union that has neither, which is what made ``SemiCRFRecipe`` rebuild a decoder
    # from values the checker could not tell apart from a tensor.
    scorer: SpanScorer

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

        The module is the authority on placement, not the evidence: numpy ``Frames`` have
        no device of their own, and the DP adds its evidence straight to the
        parameters. Sourcing it here lets a CUDA-resident decoder score CPU-born
        evidence."""
        return next(self.scorer.parameters()).device

    def _prepare(
        self, frames_list: Sequence[Frames]
    ) -> tuple[Evidence, Grid]:
        """Evidence for a batch, and the ``Grid`` its tracks share.

        All tracks come off one feature hop, so the grid is derived once from the first
        rather than per track."""
        ev = build_evidence(frames_list, self.stride, self.device)
        return ev, self.scorer.grid(ev.frame_times[0])

    def nll(self, frames: Frames, notes: list[NoteEvent]) -> torch.Tensor:
        """``log Z - score(gold)`` for one track -- the batch of one."""
        return self.nll_batch([frames], [notes])[0]

    def nll_batch(
        self,
        frames_list: Sequence[Frames],
        notes_list: Sequence[list[NoteEvent]],
    ) -> torch.Tensor:
        """Per-track NLL ``(B,)``: one batched ``log Z`` scan over the padded evidence,
        minus each track's gold score.

        Both halves read the *same* padded batch: gold indices never reach past a
        track's true length, so the flat padding beyond it is never gathered."""
        ev, grid = self._prepare(frames_list)
        golds = [
            gold_segments(ft, n, grid.dmax, self.scorer.labels)
            for ft, n in zip(ev.frame_times, notes_list, strict=True)
        ]
        return log_partitions(self.scorer, ev, grid) - segmentation_scores(
            self.scorer, ev, golds, grid
        )

    def nll_chunk(
        self,
        frames_list: Sequence[Frames],
        golds: list[list[tuple[int, int, int]]],
        carry: torch.Tensor | None,
        context: int,
        right: int = 0,
        prevs: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One truncated-BPTT step: ``(A, gold_score, window)`` for a batch of chunks.

        Deliberately *not* an NLL: ``A`` is the cumulative log-mass over everything
        scanned so far, so the caller forms the per-chunk loss by telescoping,
        ``(A_k - A_{k-1}) - gold_score_k``.

        ``golds`` are *window-local* segmentations supplied by the dataset rather than
        derived from notes here: gold is built once over the whole track and split by
        chunk, which is what makes the pieces tile it exactly and the losses telescope.
        Deriving it per chunk would re-run the audible-gap rule on a truncated view and
        silently disagree at the seams.

        ``prevs`` is the label preceding each row's first gold segment -- what the
        previous chunk ended on. ``None`` means ``START`` for every row, which is true
        only of a track's first chunk: ``log Z`` applies the real carried transition, so
        assuming ``START`` here makes the two halves of the NLL score different
        quantities."""
        ev, grid = self._prepare(frames_list)
        A, window = chunk_log_partition(self.scorer, ev, grid, carry, context, right)
        gold = segmentation_scores(self.scorer, ev, golds, grid, prevs)
        return A, gold, window

    def decode(
        self, frames: Frames, note_bias: float = 0.0, with_confidence: bool = False
    ) -> list[NoteEvent]:
        """The notes implied by ``frames`` under the learned scorer (Viterbi).

        ``with_confidence`` attaches each note's posterior, ``P(span | x)``, which
        Viterbi itself cannot report -- it returns the best path and says nothing about
        how close the runner-up came, so every note arrives equally certain. Off by
        default because it costs a second scan with its autograd graph retained,
        roughly 3-4x the decode: worth it for a person reviewing a recording, waste for
        a validation pass that only wants F1.

        ``note_bias`` is a **diagnostic**, not a setting: it measures how far the
        best single path sits from the distribution's own expected note count, which
        is what said the collapse was a readout and not a worse model. Deliberately a
        call-site argument that nothing persists -- the pipeline decodes uncorrected,
        so no result can silently acquire a constant fitted to somebody's val set."""
        ev, grid = self._prepare([frames])
        if ev.lengths[0] == 0:
            return []
        segs = viterbi(self.scorer, ev, grid, note_bias)
        confidence = (
            segment_confidence(span_marginals(self.scorer, ev, grid), segs, grid)
            if with_confidence else None
        )
        return self._segs_to_notes(segs, ev, confidence)

    def _segs_to_notes(
        self,
        segs: list[tuple[int, int, int]],
        ev: Evidence,
        confidence: Mapping[tuple[int, int, int], float] | None = None,
    ) -> list[NoteEvent]:
        """Turn ``note`` segments into ``NoteEvent``s, from whichever pitch source the
        scorer's alphabet implies.

        **Under a pitch alphabet the label already is the answer.** The segment was
        scored as ``note@p`` against the pitch head's own log-probability of ``p`` across
        it, so reading a median off the contour afterwards would discard the decision the
        DP just made -- and could name a pitch the segmentation was never scored under.

        Under ``BINARY`` the label says only *note*, so the pitch is the median contour
        pitch over the span, and a span the contour left entirely pitchless yields no
        note. That contour is pyin's or the model's own and this cannot tell:
        ``LearnedEncoder`` substitutes the pitch head's ``decode_contour`` output when the
        model has that head, so both arrive as a plain ``Contour``."""
        labels = self.scorer.labels
        ft = ev.frame_times[0]
        hop = float(np.median(np.diff(ft))) if ft.size > 1 else 0.01 * self.stride
        pitch_of: Callable[[int, float, float], float | None]
        if labels.names_pitch:
            def pitch_of(label: int, onset: float, offset: float) -> float | None:
                return float(labels.midi_of(label))
        else:
            if ev.contours is None:
                raise ValueError(
                    "cannot name note pitches: this scorer's alphabet does not name "
                    "them and this Evidence carries no contour. The DP scores spans "
                    "without one, but a NoteEvent needs a pitch -- decode through "
                    "Frames from an encoder (pyin or a pitch head), not a contour-less "
                    "training batch."
                )
            contour = ev.contours[0]
            ct, cf = contour.times, contour.freqs
            cmidi = np.full(cf.shape, np.nan)
            cmidi[cf > 0] = librosa.hz_to_midi(cf[cf > 0])

            def pitch_of(label: int, onset: float, offset: float) -> float | None:
                sel = (ct >= onset) & (ct < offset)
                if not sel.any() or np.isnan(cmidi[sel]).all():
                    return None
                median = float(np.nanmedian(cmidi[sel]))
                return None if np.isnan(median) else median

        notes: list[NoteEvent] = []
        for label, i, j in segs:
            if not labels.is_note(label):
                continue
            onset, offset = float(ft[i]), float(ft[j - 1] + hop)
            pitch = pitch_of(label, onset, offset)
            if pitch is None:
                continue
            notes.append(NoteEvent(
                onset=onset, offset=offset, midi_pitch=pitch,
                # 1.0 when nobody asked -- ``NoteEvent``'s own default, and the
                # honest value for a decode that computed no posterior. Reporting
                # a made-up number would be worse than reporting none, because a
                # reviewer sorting by confidence would be sorting noise.
                confidence=1.0 if confidence is None else confidence.get((label, i, j), 0.0),
            ))
        return notes
