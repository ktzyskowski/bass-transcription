"""``SemiCRFDecoder``: a learned, global note decoder over per-frame ``Cues``.

Where ``HeuristicDecoder`` greedily peak-picks the onset curve and thresholds the
voicing curve, the semi-Markov CRF decodes the *whole* track at once: it finds the
single highest-scoring segmentation of the frame grid into ``note`` and ``rest``
spans under a handful of learned potentials. There is no merge pass -- a learned
duration potential makes fragmentation globally costly, so the over-segmentation the
heuristic repairs after the fact is instead priced into the objective and never
produced.

Two faces, one recurrence (the segmental scan ``alpha[j] = f(alpha[i<j])``):

  - ``decode(cues) -> notes`` runs segmental **Viterbi** (the max semiring) for the
    single best segmentation, then reads a pitch off ``cues.contour`` per ``note``
    span. This is the ``Decoder`` interface.
  - ``nll(cues, notes) -> loss`` runs the segmental **forward** algorithm (the
    logsumexp semiring) for ``log Z``, and scores the gold segmentation built from
    the reference notes; ``log Z - gold`` is the negative log-likelihood, fully
    differentiable, so the potentials -- and, end to end, the encoder feeding them --
    train by gradient descent. See ``docs/semi-crf.md``.

The CRF consumes the *soft* cues directly (onset logit, per-frame log P(voiced)),
which is how it can keep a sub-threshold onset the heuristic's peak-pick would drop.
Three learned unary signals per ``note`` span -- accumulated onset evidence, voicing
mass, and a penalty per interior pyin pitch-change (the legato split cue) -- plus a
duration table, biases, and a 2x2 transition block. Pitch is never invented: it is
read from ``cues.contour`` after decoding, exactly as the heuristic does.

The frame grid is decimated by ``STRIDE`` before the DP (the note boundaries the
±50 ms tolerance cares about survive an ~35 ms grid, and the scan cost is quadratic
in the segment cap), and gold rests come only from audible gaps -- a MIDI note-off
gap shorter than ``GAP_REST_S`` is an artifact the audio rings through, so the
previous note is extended to close it rather than teaching the model a rest the
audio never contains (the under-segmentation trap; see ``[[slakh-label-quirks]]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch
from torch import nn

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.base import Decoder
from stemify.decoders.frames import resample_mask
from stemify.decoders.heuristic import PITCH_SPLIT_SEMITONES
from stemify.notes import NoteEvent

# Decimate the ~11.6 ms feature grid before the DP: the scan is O(T * dmax), so a
# coarser grid is a direct speed lever, and a ~35 ms boundary resolution is well
# inside the 50 ms onset tolerance the note metric scores at.
STRIDE = 3
# Longest representable segment. Must exceed the longest real bass note, or a held
# note is forced to split; 4 s is comfortably past that.
DMAX_S = 4.0
# A between-note MIDI gap shorter than this is a note-off artifact (the audio rings
# on), not audible silence -- gold closes it into a contiguous note->note transition
# instead of a phantom rest. Longer gaps stay rests. This is the fix that stops the
# CRF learning to avoid note->note and collapsing legato runs.
GAP_REST_S = 0.12
# Note-duration potential buckets, in seconds. A learned scalar per bucket lets the
# model say "notes this short are implausible" -- the over-segmentation lever.
DUR_EDGES_S = np.array([0.06, 0.12, 0.20, 0.35, 0.6, 1.0, 2.0])

# Segment labels. START is only reachable at position 0, which makes the first-
# segment case fall out of the same scan as every other step (no special-casing).
NOTE, REST, START = 0, 1, 2
_NEG = -1e9  # log 0, for unreachable states


# --------------------------------------------------------------------------- #
# emissions: Cues -> the per-frame tensors the DP scans                       #
# --------------------------------------------------------------------------- #
@dataclass
class _Emissions:
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


def _pitch_jumps(contour: Contour, frame_times: np.ndarray) -> np.ndarray:
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


def _voiced_logprobs(cues: Cues) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ``(log P(voiced), log P(unvoiced))`` on ``cues.frame_times``.

    Prefer the voicing head; with none, fall back to pyin's own voiced mask
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability --
    the CRF still runs on DSP-only cues, just without a learned rest signal."""
    if cues.voicing is not None:
        vp = np.clip(cues.voicing, 1e-6, 1 - 1e-6)
    else:
        mask = resample_mask(
            cues.contour.times, cues.contour.freqs > 0, cues.frame_times
        )
        vp = np.clip(mask, 1e-6, 1 - 1e-6)
    return np.log(vp), np.log(1 - vp)


def _emissions(cues: Cues, stride: int) -> _Emissions:
    """Build the strided ``_Emissions`` the DP scans from soft ``Cues``."""
    op = np.clip(cues.onset[::stride], 1e-6, 1 - 1e-6)
    onset_logit = torch.from_numpy(np.log(op / (1 - op)).astype(np.float32))
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])

    vlp_full, ulp_full = _voiced_logprobs(cues)
    vlp = torch.from_numpy(np.ascontiguousarray(vlp_full[::stride]).astype(np.float32))
    ulp = torch.from_numpy(np.ascontiguousarray(ulp_full[::stride]).astype(np.float32))
    Vc = torch.cat([torch.zeros(1), torch.cumsum(vlp, 0)])
    Uc = torch.cat([torch.zeros(1), torch.cumsum(ulp, 0)])

    jump = _pitch_jumps(cues.contour, frame_times)
    Jc = torch.cat([torch.zeros(1), torch.cumsum(torch.from_numpy(jump), 0)])
    return _Emissions(onset_logit, Vc, Uc, Jc, frame_times, cues.contour)


def _gold_segments(
    frame_times: np.ndarray, ref: list[NoteEvent], dmax: int
) -> list[tuple[int, int, int]]:
    """Tile ``[0, T)`` into ``(label, i, j)`` gold segments from reference notes.

    Each note is a ``note`` span (clipped to ``dmax``); the space between notes is a
    ``rest`` only when the gap is longer than ``GAP_REST_S`` -- real audible silence.
    A shorter gap is a MIDI note-off artifact the audio rings through, so the previous
    note is extended to close it, yielding contiguous ``note->note`` gold. Training on
    the raw MIDI gaps instead teaches rests the audio never contains and a spurious
    ``note->note`` penalty that collapses legato runs (the under-segmentation trap)."""
    T = frame_times.size
    spans = []
    for n in sorted(ref, key=lambda n: n.onset):
        i = int(np.clip(np.searchsorted(frame_times, n.onset), 0, T - 1))
        j = int(np.clip(np.searchsorted(frame_times, n.offset), i + 1, T))
        spans.append((i, j))

    segs: list[tuple[int, int, int]] = []

    def add_rest(lo: int, hi: int) -> None:
        while lo < hi:
            k = min(hi, lo + dmax)
            segs.append((REST, lo, k))
            lo = k

    cur = 0
    for i, j in spans:
        i = max(i, cur)
        if j <= i:
            continue
        if i > cur:  # a gap: audible rest, or a note-off artifact to close?
            gap_s = float(frame_times[i] - frame_times[cur]) if cur < T else GAP_REST_S + 1
            if gap_s > GAP_REST_S:
                add_rest(cur, i)
            elif segs and segs[-1][0] == NOTE:  # extend prev note over the artifact
                pi, pj = segs[-1][1], min(i, segs[-1][1] + dmax)
                segs[-1] = (NOTE, pi, pj)
                if pj < i:  # dmax-limited: remainder must still tile
                    add_rest(pj, i)
            else:
                add_rest(cur, i)  # no prior note to extend
        j = min(j, i + dmax)  # clip an over-long note to one representable segment
        segs.append((NOTE, i, j))
        cur = j
    if cur < T:
        add_rest(cur, T)
    return segs


# --------------------------------------------------------------------------- #
# the decoder                                                                 #
# --------------------------------------------------------------------------- #
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
        self.dmax_s = dmax_s
        self.register_buffer("dur_edges_s", torch.as_tensor(dur_edges_s, dtype=torch.float32))

        # Learned potentials (~15 scalars). w_* weight the soft per-frame evidence;
        # the biases set a note/rest baseline; `dur` is the duration table; `trans`
        # is the note/rest 2x2 block and `init` is START -> {note, rest}.
        self.w_on = nn.Parameter(torch.tensor(1.0))
        self.w_v = nn.Parameter(torch.tensor(1.0))
        self.w_u = nn.Parameter(torch.tensor(1.0))
        self.w_pitch = nn.Parameter(torch.tensor(0.0))  # per interior pitch-change
        self.b_note = nn.Parameter(torch.tensor(0.0))
        self.b_rest = nn.Parameter(torch.tensor(0.0))
        self.dur = nn.Parameter(torch.zeros(len(dur_edges_s) + 1))
        self.trans = nn.Parameter(torch.zeros(2, 2))  # [prev, next] over {note, rest}
        self.init = nn.Parameter(torch.zeros(2))       # START -> {note, rest}

    # ---- grid-dependent shapes -------------------------------------------- #
    def _grid(self, frame_times: np.ndarray) -> tuple[int, torch.Tensor]:
        """``(dmax_frames, dur_bucket)`` for a frame grid: the max segment length in
        frames and the length->bucket map indexing ``self.dur``. Derived from the
        grid's hop rather than stored, so the decoder is agnostic to the exact hop."""
        hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
        dmax = max(1, round(self.dmax_s / hop))
        edges = np.round(self.dur_edges_s.numpy() / hop).astype(int)
        dur_bucket = torch.from_numpy(
            np.searchsorted(edges, np.arange(dmax + 1)).astype(np.int64)
        )
        return dmax, dur_bucket

    def _trans_to(self) -> torch.Tensor:
        """(3, 2): rows = prev in {note, rest, START}, cols = next in {note, rest}."""
        return torch.stack([self.trans[NOTE], self.trans[REST], self.init])

    # ---- unary segment potentials ----------------------------------------- #
    def _note_potentials(
        self, e: _Emissions, idx: torch.Tensor, j: int, dur_bucket: torch.Tensor
    ) -> torch.Tensor:
        """``note``-potential of every span ``[i, j)`` for ``i in idx``: onset
        evidence at ``i``, voicing mass, interior pitch-change penalty, bias and the
        duration term -- vectorised over the candidate starts."""
        d = j - idx
        return (
            self.w_on * e.onset_logit[idx]
            + self.w_v * (e.Vc[j] - e.Vc[idx])
            + self.w_pitch * (e.Jc[j] - e.Jc[idx + 1])  # interior changes only
            + self.b_note
            + self.dur[dur_bucket[d]]
        )

    def _rest_potentials(self, e: _Emissions, idx: torch.Tensor, j: int) -> torch.Tensor:
        return self.w_u * (e.Uc[j] - e.Uc[idx]) + self.b_rest

    # ---- segmental forward (log Z) and gold, for the NLL ------------------ #
    def _log_partition(self, e: _Emissions, dmax: int, dur_bucket: torch.Tensor) -> torch.Tensor:
        """``log Z``: the logsumexp over every segmentation, via the segmental
        forward recurrence. ``alpha[j][label]`` is the mass of all ways to tile
        ``[0, j)`` ending in a ``label`` span; differentiable, so ``d log Z`` gives
        the model's expected segment features."""
        T = e.T
        trans_to = self._trans_to()  # (3, 2)
        a0 = e.onset_logit.new_full((3,), _NEG)
        a0[START] = 0.0
        alpha = [a0]
        for j in range(1, T + 1):
            lo = max(0, j - dmax)
            idx = torch.arange(lo, j)
            note = self._note_potentials(e, idx, j, dur_bucket)
            rest = self._rest_potentials(e, idx, j)
            seg = torch.stack([note, rest], dim=-1)          # (W, 2=next)
            prev = torch.stack(alpha[lo:j], dim=0)           # (W, 3=prev)
            term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
            val = torch.logsumexp(term.reshape(-1, 2), dim=0)  # (2,)
            alpha.append(torch.cat([val, val.new_full((1,), _NEG)]))
        return torch.logsumexp(alpha[T][:2], dim=0)

    def _gold_score(
        self, e: _Emissions, gold: list[tuple[int, int, int]], dur_bucket: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalised score of the gold segmentation: the sum of its segments'
        unary potentials and the transitions between them."""
        score = e.onset_logit.new_zeros(())
        prev = START
        for label, i, j in gold:
            idx = torch.tensor([i])
            if label == NOTE:
                unary = self._note_potentials(e, idx, j, dur_bucket)[0]
            else:
                unary = self._rest_potentials(e, idx, j)[0]
            trans = self.init[label] if prev == START else self.trans[prev, label]
            score = score + unary + trans
            prev = label
        return score

    def nll(self, cues: Cues, notes: list[NoteEvent]) -> torch.Tensor:
        """Negative log-likelihood of the gold segmentation (from ``notes``) under
        the model, given ``cues``: ``log Z - score(gold)``. The differentiable
        training objective -- backprop reaches the encoder that produced ``cues``."""
        e = _emissions(cues, self.stride)
        dmax, dur_bucket = self._grid(e.frame_times)
        gold = _gold_segments(e.frame_times, notes, dmax)
        return self._log_partition(e, dmax, dur_bucket) - self._gold_score(e, gold, dur_bucket)

    # ---- segmental Viterbi decode ----------------------------------------- #
    @torch.no_grad()
    def _viterbi(self, e: _Emissions, dmax: int, dur_bucket: torch.Tensor) -> list[tuple[int, int, int]]:
        """The single highest-scoring segmentation: the same recurrence as the
        forward pass with max in place of logsumexp, plus backpointers to trace the
        argmax path from the terminal state back through START."""
        T = e.T
        trans_to = self._trans_to()  # (3, 2)
        a0 = e.onset_logit.new_full((3,), _NEG)
        a0[START] = 0.0
        alpha = [a0]
        back: list[dict | None] = [None]
        for j in range(1, T + 1):
            lo = max(0, j - dmax)
            idx = torch.arange(lo, j)
            note = self._note_potentials(e, idx, j, dur_bucket)
            rest = self._rest_potentials(e, idx, j)
            seg = torch.stack([note, rest], dim=-1)          # (W, 2)
            prev = torch.stack(alpha[lo:j], dim=0)           # (W, 3)
            term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
            flat = term.reshape(-1, 2)                        # (W*3, 2)
            val, argf = flat.max(dim=0)                       # (2,), (2,)
            alpha.append(torch.cat([val, val.new_full((1,), _NEG)]))
            back.append(
                {lab: (lo + int(argf[lab]) // 3, int(argf[lab]) % 3) for lab in (NOTE, REST)}
            )
        label = int(torch.stack([alpha[T][NOTE], alpha[T][REST]]).argmax())
        j, segs = T, []
        while j > 0:
            i, prev = back[j][label]
            segs.append((label, i, j))
            j, label = i, prev
            if label == START:
                break
        segs.reverse()
        return segs

    def _segs_to_notes(self, segs: list[tuple[int, int, int]], e: _Emissions) -> list[NoteEvent]:
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

    def decode(self, cues: Cues) -> list[NoteEvent]:
        """The notes implied by ``cues`` under the learned potentials (Viterbi)."""
        e = _emissions(cues, self.stride)
        if e.T == 0:
            return []
        dmax, dur_bucket = self._grid(e.frame_times)
        return self._segs_to_notes(self._viterbi(e, dmax, dur_bucket), e)
