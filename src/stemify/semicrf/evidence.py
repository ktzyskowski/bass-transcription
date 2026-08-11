"""``Frames`` -> the per-frame tensors the DP scans, and how spans read them.

``Evidence`` is **always batched**: one track is a batch of one. There is no separate
single-track type, because the two differed only in a leading axis while duplicating
every accessor and forcing the alpha window to be shape-polymorphic.

Voicing and the pitch cues enter as *prefix cumsums*, so a span's mass is an O(1)
difference ``C[j] - C[i]`` rather than a re-sum per candidate. That is what keeps the
scan linear in the segment cap, and it is the constraint every future feature has to
satisfy -- an interior term must be additive over frames, or endpoint- and length-only.

The span accessors live here rather than in the scorer because they encode *index
conventions*, not modelling choices.

**The DP reads only learned evidence.** A contour is still read to name a decoded
note's pitch under a ``{note, rest}`` alphabet, and is optional there; no potential
reads it.

``torch.as_tensor`` is the load-bearing trick throughout: it treats a numpy field as
a constant (inference/decode) and a grad-bearing one as a graph node (training), so
one builder serves both and the DP is backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from stemify.music.contour import Contour
from stemify.transcription.frames import Frames
from stemify.audio.features import BINS_PER_SEMITONE
from stemify.semicrf.cumsums import pitch_logprob_cumsum, voiced_logcumsum
from stemify.semicrf.placement import (
    check_one_device, frames_device, on_device,
)

# A span index: an int for a single span, or a tensor when the scan broadcasts a
# whole row of candidate starts at once. Every accessor takes all three the same
# way, which is what lets one `SpanScorer.unary` serve the scan and the gold scorer.
Idx = torch.Tensor | int




@dataclass
class Evidence:
    """The strided per-frame evidence for a batch of tracks, as the DP consumes it.

    **Device invariant: every tensor field lives on one device.** The DP adds them
    term by term, so a split is a hard error the moment the encoder is not on CPU.
    The fields descend from ``Frames``' grad-bearing tensors and follow the encoder
    onto its accelerator; numpy ``Frames`` is placed explicitly by ``build_evidence``.

    Ragged tracks are padded to a common length: ``onset_logit`` pads with 0 and the
    cumsums pad **flat** (last value repeated), so no voiced or onset mass accrues past
    a track's true end. The scan reads each track out at its own ``lengths[b]``.
    """

    onset_logit: torch.Tensor       # (B, Tmax)
    voiced_c: torch.Tensor          # (B, Tmax+1) cumsum log P(voiced)
    rest_c: torch.Tensor            # (B, Tmax+1) cumsum log(1 - P(voiced))
    lengths: list[int]
    frame_times: list[np.ndarray]   # per track, strided grid, absolute seconds
    # Per track, for naming a decoded note's pitch. None when the model's own pitch
    # head supplies it instead -- no DP potential reads this.
    contours: list[Contour] | None
    # (B, Tmax+1, n_semitones) cumsum of log P(pitch in this semitone), from a learned
    # pitch head and None when the model has none. The DP's only pitch-*valued*
    # evidence. Width comes from the head's own axis (K // BINS_PER_SEMITONE), which is
    # the same rule `LabelSet.from_bins` uses -- so a scorer's alphabet and this agree
    # by construction rather than by a constant someone keeps in step.
    pitch_logprob_c: torch.Tensor | None = None
    # (B, Tmax, C) the backbone's per-frame activations, strided. Raw rather than a
    # cumsum, and alone among these fields in that: every other entry here is a fixed
    # function of the encoder's output, so it can be summed once at build time, while
    # what a scorer *wants* pooled out of the trunk is itself learned. A neural scorer
    # therefore builds its own prefix sums over its own per-frame projection of this.
    # None from a producer with no backbone (the DSP encoder, the loss's test doubles).
    hidden: torch.Tensor | None = None
    # (B, Tmax+1, dmax, n_labels) unary scores for every span, or None.
    #
    # **A scorer-owned cache, and it lives here because an ``Evidence`` is exactly one
    # batch.** A scorer whose potential is a network cannot afford to be called once per
    # scan step -- the DP is launch-bound, and a per-step MLP is `T` launches of a few
    # hundred rows. It computes every span at once in ``precompute`` and this holds the
    # result. Keying it on the ``Evidence`` object is what makes the lifetime right
    # without a cache-invalidation rule: a new batch is a new ``Evidence``, and ``one()``
    # drops it rather than carrying a table built for a different set of rows.
    span_scores: torch.Tensor | None = None

    def __post_init__(self) -> None:
        check_one_device(
            "Evidence",
            onset_logit=self.onset_logit, voiced_c=self.voiced_c,
            rest_c=self.rest_c,
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
        def cut(x: torch.Tensor | None) -> torch.Tensor | None:
            return None if x is None else x[b : b + 1, : t + 1]
        # Keyword arguments, not positional: several of these are same-shaped
        # (B, Tmax+1) cumsums, so transposing two of them here would type-check, run,
        # train, and score the wrong quantity -- a wrong model that never raises.
        return Evidence(
            onset_logit=self.onset_logit[b : b + 1, :t],
            voiced_c=self.voiced_c[b : b + 1, : t + 1],
            rest_c=self.rest_c[b : b + 1, : t + 1],
            lengths=[t],
            frame_times=[self.frame_times[b]],
            contours=None if self.contours is None else [self.contours[b]],
            pitch_logprob_c=cut(self.pitch_logprob_c),
            # `[:, :t]`, not `cut`: `hidden` is per-frame, not a cumsum, so it has
            # T entries rather than T+1.
            hidden=None if self.hidden is None else self.hidden[b : b + 1, :t],
            # `span_scores` is deliberately *not* carried. It is indexed by row, and
            # a table built for the whole batch would silently answer row 0 for
            # every question asked of this single-row view.
        )

    # --- span accessors: the index conventions, in one place --------------------

    def mean_pitch_logprob(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """Mean per-frame log-probability of each semitone across ``[i, j)``.

        ``(..., n_semitones)`` -- one score per pitch the head can name, which is what
        lets a ``note@p`` span be worth something different from ``note@q``. Raises
        rather than zeroing when the model has no pitch head: a scorer whose alphabet
        names pitches cannot fall back to scoring them all alike, because then every
        pitch relabelling of a segmentation is equally good and ``log Z`` spreads its
        mass over all of them.

        A *mean*, so the value does not grow with span length and a long note is not
        penalised for being long -- duration is the duration potential's job.
        """
        if self.pitch_logprob_c is None:
            raise ValueError(
                "no pitch-valued evidence: this Evidence carries no pitch head, so a "
                "pitch-labelled alphabet has nothing to score. Use a {note, rest} "
                "scorer, or decode through cues from a model with a pitch head."
            )
        span = j - i
        d = span if isinstance(span, int) else span.unsqueeze(-1)
        return (self.pitch_logprob_c[b, j] - self.pitch_logprob_c[b, i]) / d


@dataclass(frozen=True)
class _TrackTensors:
    """One track's per-frame tensors, before padding into a batch.

    Named fields rather than the six-tuple this was, which the padding loop indexed as
    ``t[0]`` through ``t[5]`` and tested for presence as ``t[4] is not None``. The
    fields carry no coordinate ambiguity -- they are all on the strided grid -- but
    positional access across sixty lines is its own way to read the wrong one.
    """

    onset_logit: torch.Tensor
    voiced_c: torch.Tensor
    rest_c: torch.Tensor
    frame_times: np.ndarray
    pitch_logprob: torch.Tensor | None
    hidden: torch.Tensor | None

    @property
    def length(self) -> int:
        """DP frames in this track -- what the batch pads up to ``Tmax``."""
        return int(self.onset_logit.shape[0])


def _track_tensors(cues: Frames, stride: int, device: torch.device) -> _TrackTensors:
    """One track's per-frame tensors, the pitch cumsum included when it has a head."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])
    # Stride first, place second: the slice is cheap on the source device and only the
    # decimated grid crosses the bus.
    onset = on_device(torch.as_tensor(cues.onset).float()[::stride], device)
    onset_logit = torch.logit(onset.clamp(1e-6, 1 - 1e-6))
    voicing = cues.voicing[::stride] if cues.voicing is not None else None
    Vc, Uc = voiced_logcumsum(voicing, cues.contour, frame_times, device)
    # A plain attribute, not a getattr default: both cue types carry this field, and
    # the probe that used to stand here is what let the numpy path silently arrive
    # without one -- zeroing the learned span features at decode.
    logits = cues.pitch_logits
    pitch_logprob = None
    if logits is not None:
        strided = on_device(logits, device)[::stride]
        n_semitones = strided.shape[-1] // BINS_PER_SEMITONE
        pitch_logprob = pitch_logprob_cumsum(strided, n_semitones)
    hidden = None if cues.hidden is None else on_device(cues.hidden, device)[::stride]
    return _TrackTensors(onset_logit, Vc, Uc, frame_times, pitch_logprob, hidden)


def build_evidence(
    cues: Frames | Sequence[Frames],
    stride: int,
    device: torch.device | str | None = None,
) -> Evidence:
    """Build the strided ``Evidence`` the DP scans, from one track or a list.

    ``device`` is the single placement every field is built onto, establishing the
    one-device invariant. Callers that own the scorer pass *its* module's device, so a
    numpy-born field follows the parameters it will be added to -- which is what makes
    numpy ``Frames`` decodable by a module on an accelerator.

    Padding is grad-preserving: the per-track tensors are scattered into buffers
    allocated off the first track."""
    # `Sequence`, not `list`: the parameter is widened so a `list[Frames]` from the
    # training loop is accepted (a `list` is invariant in its element type, so it is not
    # a `list[Frames]`). Dispatching on `list` while accepting `Sequence`
    # would send a tuple of tracks down the single-track branch.
    batch = list(cues) if isinstance(cues, Sequence) else [cues]
    if not batch:
        raise ValueError("build_evidence needs at least one track")
    device = frames_device(batch[0]) if device is None else torch.device(device)

    built = [_track_tensors(c, stride, device) for c in batch]
    _require_one_model(built)
    padded = _pad(built)

    # A batch of contours only when every track has one: a partial list would silently
    # give some decoded notes a pyin pitch and others none.
    maybe = [c.contour for c in batch]
    contours = None if any(c is None for c in maybe) else cast(list[Contour], maybe)
    return Evidence(
        onset_logit=padded.onset_logit,
        voiced_c=padded.voiced_c,
        rest_c=padded.rest_c,
        lengths=[t.length for t in built],
        frame_times=[t.frame_times for t in built],
        contours=contours,
        pitch_logprob_c=padded.pitch_logprob_c,
        hidden=padded.hidden,
    )


def _require_one_model(built: list[_TrackTensors]) -> None:
    """Every track in a batch must come from the same model.

    Raised here, once, rather than guarded at each scatter site: a batch that mixes
    these produces a wrong number rather than a crash, so the check has to be the thing
    that stops it.
    """
    if len({t.pitch_logprob is not None for t in built}) > 1:
        raise ValueError(
            "batch mixes tracks with and without pitch logits; every track in a batch "
            "must come from the same model, or a pitch label is scored against real "
            "evidence for some rows and against nothing for others"
        )
    if len({t.hidden is not None for t in built}) > 1:
        raise ValueError(
            "batch mixes tracks with and without backbone activations; a scorer that "
            "pools the trunk would read a learned projection of real features for some "
            "rows and of padding for others"
        )


@dataclass(frozen=True)
class _Padded:
    """The ``(B, Tmax, ...)`` buffers a batch of tracks scatters into.

    A dataclass rather than the ``dict[str, Tensor | None]`` this first was: splatting
    such a dict into ``Evidence(**padded)`` erases every argument's type at the call,
    which is the failure mode this codebase already names elsewhere -- a field renamed
    on one side and silently unfilled on the other.
    """

    onset_logit: torch.Tensor
    voiced_c: torch.Tensor
    rest_c: torch.Tensor
    pitch_logprob_c: torch.Tensor | None = None
    hidden: torch.Tensor | None = None


def _pad(built: list[_TrackTensors]) -> _Padded:
    """Scatter per-track tensors into ``(B, Tmax, ...)`` buffers.

    Grad-preserving: the buffers are allocated off the first track and written into, so
    autograd reaches the per-track tensors through the scatter.

    **Padded flat, not with zeros.** Every buffer here but ``onset_logit`` is a prefix
    sum, and a span's feature is a *difference* of two of them -- so a zero tail would
    make every span reaching past a track's end read a discontinuity rather than
    nothing. Repeating the last real value keeps those differences zero. ``hidden``
    pads the same way for a subtler reason: a scorer builds its own prefix sums over a
    learned projection of it, and a projection has a bias, so zero frames past the end
    would still accrue mass into every prefix.
    """
    lengths = [t.length for t in built]
    B, Tmax = len(built), max(lengths)
    ref = built[0].onset_logit
    first = built[0]

    onset = ref.new_zeros(B, Tmax)
    voiced_c = ref.new_zeros(B, Tmax + 1)
    rest_c = ref.new_zeros(B, Tmax + 1)
    hidden = (
        None if first.hidden is None
        else ref.new_zeros(B, Tmax, first.hidden.shape[-1])
    )
    logprob_c = (
        None if first.pitch_logprob is None
        else ref.new_zeros(B, Tmax + 1, first.pitch_logprob.shape[-1])
    )

    for b, track in enumerate(built):
        t = track.length
        onset[b, :t] = track.onset_logit
        _flat_pad(voiced_c, b, t, track.voiced_c)
        _flat_pad(rest_c, b, t, track.rest_c)
        if hidden is not None and track.hidden is not None:
            hidden[b, :t], hidden[b, t:] = track.hidden, track.hidden[-1]
        if track.pitch_logprob is not None:
            _flat_pad(logprob_c, b, t, track.pitch_logprob)

    return _Padded(
        onset_logit=onset, voiced_c=voiced_c, rest_c=rest_c,
        pitch_logprob_c=logprob_c, hidden=hidden,
    )


def _flat_pad(
    buffer: torch.Tensor | None, row: int, length: int, values: torch.Tensor
) -> None:
    """Write one track's prefix sums into ``buffer[row]``, repeating the last value.

    A prefix sum has ``length + 1`` entries -- one per boundary, not one per frame --
    which is why the slice ends at ``length + 1`` and the flat tail begins there.
    """
    assert buffer is not None            # allocated together with the field it holds
    buffer[row, : length + 1] = values
    buffer[row, length + 1 :] = values[-1]
