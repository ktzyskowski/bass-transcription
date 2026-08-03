"""``Cues | TensorCues`` -> the per-frame tensors the DP scans, and how spans read them.

``Evidence`` is **always batched**: one track is a batch of one. There is no separate
single-track type, because the two differed only in a leading axis while duplicating
every accessor and forcing the alpha window to be shape-polymorphic.

Voicing and the pitch cues enter as *prefix cumsums*, so a span's mass is an O(1)
difference ``C[j] - C[i]`` rather than a re-sum per candidate. That is what keeps the
scan linear in the segment cap, and it is the constraint every future feature has to
satisfy -- an interior term must be additive over frames, or endpoint- and length-only.

The span accessors live here rather than in the scorer because they encode *index
conventions*, not modelling choices.

**The DP reads only learned evidence.** It used to also score ``interior_jumps``, a
count of pyin pitch-change flags inside a span. That was deleted: ``path_length`` asks
the same question -- how far pitch moved inside this span -- of the model's own pitch
head, at the CQT grid rather than pyin's coarser one, and with a gradient. Keeping both
meant the contour was a hard dependency of the *model*, not just of the data pipeline.
A contour is still read to name a decoded note's pitch, and is optional there.

``torch.as_tensor`` is the load-bearing trick throughout: it treats a numpy ``Cues``
field as a constant (inference/decode) and a ``TensorCues`` tensor as a graph node
(training), so one builder serves both and the DP is backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.grids import resample_mask

# A span index: an int for a single span, or a tensor when the scan broadcasts a
# whole row of candidate starts at once. Every accessor takes all three the same
# way, which is what lets one `SpanScorer.unary` serve the scan and the gold scorer.
Idx = torch.Tensor | int


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

    # Pitch track (numpy) -- used to name a decoded note's pitch, never scored by the
    # DP. None when the model's own pitch head supplies it instead.
    contour: Contour | None
    onset: torch.Tensor            # (T,) P(onset), differentiable
    frame_times: np.ndarray        # (T,) feature-grid times, seconds
    voicing: torch.Tensor | None = None  # (T,) P(voiced), differentiable, or None
    # (T, n_bins) raw per-bin logits from a FrequencyHead, or None when the model has
    # no pitch head. Raw rather than softmaxed, because the span features normalise
    # across the bin axis themselves and squashing twice would be the wrong
    # distribution. Grad-bearing: this is how the CRF objective reaches the pitch head.
    pitch_logits: torch.Tensor | None = None

    @classmethod
    def from_logits(
        cls,
        contour: Contour | None,
        onset_logit: torch.Tensor,
        frame_times: np.ndarray,
        voicing_logit: torch.Tensor | None = None,
        pitch_logits: torch.Tensor | None = None,
    ) -> TensorCues:
        """Wrap a model's raw head logits, squashing the *binary* heads to the
        probabilities ``Cues`` carries so the two intermediaries hold the same quantity
        in the same range. Pitch stays as logits -- see the field."""
        voicing = None if voicing_logit is None else torch.sigmoid(voicing_logit)
        return cls(
            contour, torch.sigmoid(onset_logit), frame_times, voicing, pitch_logits
        )


@dataclass
class Evidence:
    """The strided per-frame evidence for a batch of tracks, as the DP consumes it.

    **Device invariant: every tensor field lives on one device.** The DP adds them
    term by term, so a split is a hard error the moment the encoder is not on CPU.
    The fields descend from ``TensorCues``' grad-bearing tensors and follow the encoder
    onto its accelerator; a numpy ``Cues`` is placed explicitly by ``build_evidence``.

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
    # From a learned pitch head, all None when the model has none. Three cumsums
    # because the two span features they serve need different aggregates: a *path
    # length* (how far the distribution travelled, frame to frame) and a *spread* (how
    # much the span's frames disagree with each other). See docs/semi-crf.md.
    path_c: torch.Tensor | None = None      # (B, Tmax+1) cumsum of 1 - BC(p_t, p_t-1)
    entropy_c: torch.Tensor | None = None   # (B, Tmax+1) cumsum of H(p_t)
    pitch_c: torch.Tensor | None = None     # (B, Tmax+1, K) cumsum of p_t

    def __post_init__(self) -> None:
        _check_one_device(
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
        return Evidence(
            self.onset_logit[b : b + 1, :t], self.voiced_c[b : b + 1, : t + 1],
            self.rest_c[b : b + 1, : t + 1],
            [t], [self.frame_times[b]],
            None if self.contours is None else [self.contours[b]],
            cut(self.path_c), cut(self.entropy_c), cut(self.pitch_c),
        )

    # --- span accessors: the index conventions, in one place --------------------

    def onset_at(self, b: Idx, i: Idx) -> torch.Tensor:
        """Onset logit at a span's first frame."""
        return self.onset_logit[b, i]

    def voiced_mass(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """``sum of log P(voiced)`` over ``[i, j)``."""
        return self.voiced_c[b, j] - self.voiced_c[b, i]

    def rest_mass(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """``sum of log(1 - P(voiced))`` over ``[i, j)``."""
        return self.rest_c[b, j] - self.rest_c[b, i]

    def _zeros_like_span(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """A zero of whatever shape a span accessor would have returned.

        The pitch accessors below return this when the model has no pitch head, so a
        scorer's ``unary`` stays branch-free and a term it cannot evaluate contributes
        nothing rather than raising. That keeps one scorer able to decode both a
        pitch-bearing and a pitch-less checkpoint -- the alternative is a scorer that
        inspects the evidence, which puts a capability check in the hot loop."""
        return torch.zeros_like(self.voiced_c[b, j] - self.voiced_c[b, i])

    def path_length(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """How far the pitch distribution *travelled* strictly inside ``[i, j)``.

        A sum of per-frame moves, so it is large for a step change and also for
        vibrato that returns where it started -- it measures distance covered, not
        whether the span holds one pitch. ``pitch_spread`` is the peer that answers
        the latter.

        The ``i + 1`` makes it interior-only, and that offset is the whole point: the
        pitch move at the span's own first frame is what *starts* this note, so
        charging the span for it would penalise every correctly-placed boundary."""
        if self.path_c is None:
            return self._zeros_like_span(b, i, j)
        return self.path_c[b, j] - self.path_c[b, i + 1]

    def mean_entropy(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """Average per-frame pitch entropy over ``[i, j)`` -- the span's *aleatoric*
        uncertainty, i.e. how unsure each frame is on its own."""
        if self.entropy_c is None:
            return self._zeros_like_span(b, i, j)
        return (self.entropy_c[b, j] - self.entropy_c[b, i]) / (j - i)

    def pitch_mean(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """The span's mean pitch distribution ``(..., K)``.

        The mean of the softmaxes, not the softmax of mean logits: this is a *mixture*,
        which is what "the span's pitch content" means, and a mixture preserves the two
        modes of a span straddling a pitch change. A geometric mean would suppress
        exactly the structure the spread feature exists to detect."""
        if self.pitch_c is None:
            raise ValueError("no pitch evidence: this Evidence carries no pitch head")
        span = j - i
        d = span if isinstance(span, int) else span.unsqueeze(-1)
        return (self.pitch_c[b, j] - self.pitch_c[b, i]) / d

    def pitch_spread(self, b: Idx, i: Idx, j: Idx) -> torch.Tensor:
        """Generalised Jensen-Shannon divergence of the span's frames.

        ``H(mean) - mean(H)``: total uncertainty about the span's pitch minus the
        average per-frame uncertainty, so what is left is uncertainty caused purely by
        the frames *disagreeing with each other*. Zero exactly when every frame carries
        the same distribution, and ``exp`` of it is the effective number of distinct
        pitches in the span -- ``log 2`` for a clean two-note straddle.

        Invariant to how confident the head is: a span where every frame is equally
        unsure scores 0, because ``mean(H)`` rises with ``H(mean)``. That is what
        separates *a pitch change* from *a hard-to-hear note*, which a frame-to-frame
        distance cannot do."""
        if self.pitch_c is None:
            return self._zeros_like_span(b, i, j)
        p = self.pitch_mean(b, i, j).clamp_min(1e-9)
        return -(p * p.log()).sum(-1) - self.mean_entropy(b, i, j)


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
        v = _on(voicing, device).float().clamp(1e-6, 1 - 1e-6)
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
        vlp = _on(np.log(vp).astype(np.float32), device)
        ulp = _on(np.log1p(-vp).astype(np.float32), device)
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
    sq = p.sqrt()
    bc = (sq[1:] * sq[:-1]).sum(-1)                       # (T-1,)
    path = torch.cat([bc.new_zeros(1), 1.0 - bc])         # (T,)
    entropy = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(-1)   # (T,)
    zero_row = p.new_zeros(1, p.shape[-1])
    return (
        torch.cat([path.new_zeros(1), torch.cumsum(path, 0)]),
        torch.cat([entropy.new_zeros(1), torch.cumsum(entropy, 0)]),
        torch.cat([zero_row, torch.cumsum(p, 0)]),
    )


def _track_tensors(
    cues: Cues | TensorCues, stride: int, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """One track's per-frame tensors, pitch cumsums included when it has a pitch head."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])
    # Stride first, place second: the slice is cheap on the source device and only the
    # decimated grid crosses the bus.
    onset = _on(torch.as_tensor(cues.onset).float()[::stride], device)
    onset_logit = torch.logit(onset.clamp(1e-6, 1 - 1e-6))
    voicing = cues.voicing[::stride] if cues.voicing is not None else None
    Vc, Uc = voiced_logcumsum(voicing, cues.contour, frame_times, device)
    # A plain attribute, not a getattr default: both cue types carry this field, and
    # the probe that used to stand here is what let the numpy path silently arrive
    # without one -- zeroing the learned span features at decode.
    logits = cues.pitch_logits
    pitch = None if logits is None else pitch_cumsums(_on(logits, device)[::stride])
    return onset_logit, Vc, Uc, frame_times, pitch


def build_evidence(
    cues: Cues | TensorCues | Sequence[Cues | TensorCues],
    stride: int,
    device: torch.device | str | None = None,
) -> Evidence:
    """Build the strided ``Evidence`` the DP scans, from one track or a list.

    ``device`` is the single placement every field is built onto, establishing the
    one-device invariant. Callers that own the scorer pass *its* module's device, so a
    numpy-born field follows the parameters it will be added to -- which is what makes
    a numpy ``Cues`` decodable by a module on an accelerator.

    Padding is grad-preserving: the per-track tensors are scattered into buffers
    allocated off the first track."""
    # `Sequence`, not `list`: the parameter is widened so a `list[TensorCues]` from the
    # training loop is accepted (a `list` is invariant in its element type, so it is not
    # a `list[Cues | TensorCues]`). Dispatching on `list` while accepting `Sequence`
    # would send a tuple of tracks down the single-track branch.
    batch = list(cues) if isinstance(cues, Sequence) else [cues]
    if not batch:
        raise ValueError("build_evidence needs at least one track")
    device = _cues_device(batch[0]) if device is None else torch.device(device)

    built = [_track_tensors(c, stride, device) for c in batch]
    lengths = [int(t[0].shape[0]) for t in built]
    B, Tmax = len(built), max(lengths)

    onset = built[0][0].new_zeros(B, Tmax)
    Vc = onset.new_zeros(B, Tmax + 1)
    Uc = onset.new_zeros(B, Tmax + 1)
    # Bind the first track's pitch cumsums rather than re-indexing `built`: the
    # buffers below exist exactly when this is non-None, and writing that as one
    # name is what makes the coupling visible instead of implied by two `has_pitch`
    # reads several lines apart.
    first_pitch = built[0][4]
    if first_pitch is not None and any(t[4] is None for t in built):
        raise ValueError(
            "batch mixes tracks with and without pitch logits; every track in a batch "
            "must come from the same model, or the span features silently read zeros "
            "for some rows and real evidence for others"
        )
    Pc = Hc = Dc = None
    if first_pitch is not None:
        K = first_pitch[2].shape[-1]
        Pc = onset.new_zeros(B, Tmax + 1)
        Hc = onset.new_zeros(B, Tmax + 1)
        Dc = onset.new_zeros(B, Tmax + 1, K)

    for b, (o, v, u, _, pitch) in enumerate(built):
        t = lengths[b]
        onset[b, :t] = o
        Vc[b, : t + 1], Vc[b, t + 1 :] = v, v[-1]  # flat past the end
        Uc[b, : t + 1], Uc[b, t + 1 :] = u, u[-1]
        if pitch is None or Pc is None or Hc is None or Dc is None:
            continue
        pc, hc, dc = pitch
        Pc[b, : t + 1], Pc[b, t + 1 :] = pc, pc[-1]
        Hc[b, : t + 1], Hc[b, t + 1 :] = hc, hc[-1]
        Dc[b, : t + 1], Dc[b, t + 1 :] = dc, dc[-1]

    # A batch of contours only when every track has one: a partial list would silently
    # give some decoded notes a pyin pitch and others none.
    maybe = [c.contour for c in batch]
    contours = None if any(c is None for c in maybe) else cast(list[Contour], maybe)
    return Evidence(
        onset, Vc, Uc, lengths, [t[3] for t in built], contours, Pc, Hc, Dc,
    )
