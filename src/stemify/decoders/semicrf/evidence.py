"""``Cues | TensorCues`` -> the per-frame tensors the DP scans, and how spans read them.

``Evidence`` is **always batched**: one track is a batch of one. There is no separate
single-track type, because the two differed only in a leading axis while duplicating
every accessor and forcing the alpha window to be shape-polymorphic.

Voicing and the pitch-change cue enter as *prefix cumsums*, so a span's mass is an
O(1) difference ``C[j] - C[i]`` rather than a re-sum per candidate. That is what keeps
the scan linear in the segment cap, and it is the constraint every future feature has
to satisfy -- an interior term must be additive over frames, or endpoint- and
length-only.

The span accessors live here rather than in the scorer because they encode *index
conventions*, not modelling choices. In particular ``interior_jumps`` owns the
``i + 1`` offset that makes a pitch change at the span's own first frame not count
against it; that offset used to be applied by three different callers, one of which
delegated it to *its* caller in a docstring.

``torch.as_tensor`` is the load-bearing trick throughout: it treats a numpy ``Cues``
field as a constant (inference/decode) and a ``TensorCues`` tensor as a graph node
(training), so one builder serves both and the DP is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.grids import resample_mask

# How far pyin's pitch must move for ``pitch_jumps`` to mark a frame. This was
# ``heuristic.PITCH_SPLIT_SEMITONES`` until that constant was turned off by default (a
# pitch change no longer *places* a note there -- see its docstring). The two usages
# were never the same question: there the threshold decided a hard note boundary, here
# it only raises a feature the CRF is free to weight, so the value is pinned locally at
# what every recorded semi-CRF run used rather than following a default that now means
# something else.
JUMP_SEMITONES = 0.8


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

    contour: Contour               # pitch track (numpy) -- used for note pitches
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
        contour: Contour,
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

    **Device invariant: all four tensor fields live on one device.** This is the type
    that needs it, because it is where two provenances meet: ``onset_logit``,
    ``voiced_c`` and ``rest_c`` descend from ``TensorCues``' grad-bearing tensors and
    follow the encoder onto its accelerator, while ``jump_c`` descends from a numpy
    contour and is CPU-born. The DP adds them term by term, so a split is a hard error
    the moment the encoder is not on CPU.

    Ragged tracks are padded to a common length: ``onset_logit`` pads with 0 and the
    cumsums pad **flat** (last value repeated), so no voiced or onset mass accrues past
    a track's true end. The scan reads each track out at its own ``lengths[b]``.
    """

    onset_logit: torch.Tensor       # (B, Tmax)
    voiced_c: torch.Tensor          # (B, Tmax+1) cumsum log P(voiced)
    rest_c: torch.Tensor            # (B, Tmax+1) cumsum log(1 - P(voiced))
    jump_c: torch.Tensor            # (B, Tmax+1) cumsum of pyin pitch-change flags
    lengths: list[int]
    frame_times: list[np.ndarray]   # per track, strided grid, absolute seconds
    contours: list[Contour]         # per track, for reading note pitches
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
            rest_c=self.rest_c, jump_c=self.jump_c,
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
        cut = lambda x, k=1: None if x is None else x[b : b + 1, : t + k]
        return Evidence(
            self.onset_logit[b : b + 1, :t], self.voiced_c[b : b + 1, : t + 1],
            self.rest_c[b : b + 1, : t + 1], self.jump_c[b : b + 1, : t + 1],
            [t], [self.frame_times[b]], [self.contours[b]],
            cut(self.path_c), cut(self.entropy_c), cut(self.pitch_c),
        )

    # --- span accessors: the index conventions, in one place --------------------

    def onset_at(self, b, i) -> torch.Tensor:
        """Onset logit at a span's first frame."""
        return self.onset_logit[b, i]

    def voiced_mass(self, b, i, j) -> torch.Tensor:
        """``sum of log P(voiced)`` over ``[i, j)``."""
        return self.voiced_c[b, j] - self.voiced_c[b, i]

    def rest_mass(self, b, i, j) -> torch.Tensor:
        """``sum of log(1 - P(voiced))`` over ``[i, j)``."""
        return self.rest_c[b, j] - self.rest_c[b, i]

    def interior_jumps(self, b, i, j) -> torch.Tensor:
        """Pitch changes landing **strictly inside** ``[i, j)``.

        The ``i + 1`` is the whole point: a pitch change at the span's own first frame
        is what *starts* this note, so charging the span for it would penalise every
        correctly-placed boundary."""
        return self.jump_c[b, j] - self.jump_c[b, i + 1]

    def _zeros_like_span(self, b, i, j) -> torch.Tensor:
        """A zero of whatever shape a span accessor would have returned.

        The pitch accessors below return this when the model has no pitch head, so a
        scorer's ``unary`` stays branch-free and a term it cannot evaluate contributes
        nothing rather than raising. That keeps one scorer able to decode both a
        pitch-bearing and a pitch-less checkpoint -- the alternative is a scorer that
        inspects the evidence, which puts a capability check in the hot loop."""
        return torch.zeros_like(self.voiced_c[b, j] - self.voiced_c[b, i])

    def path_length(self, b, i, j) -> torch.Tensor:
        """How far the pitch distribution *travelled* strictly inside ``[i, j)``.

        A sum of per-frame moves, so it is large for a step change and also for
        vibrato that returns where it started -- it measures distance covered, not
        whether the span holds one pitch. ``pitch_spread`` is the peer that answers
        the latter. Interior-only, on the same reasoning as ``interior_jumps``."""
        if self.path_c is None:
            return self._zeros_like_span(b, i, j)
        return self.path_c[b, j] - self.path_c[b, i + 1]

    def mean_entropy(self, b, i, j) -> torch.Tensor:
        """Average per-frame pitch entropy over ``[i, j)`` -- the span's *aleatoric*
        uncertainty, i.e. how unsure each frame is on its own."""
        if self.entropy_c is None:
            return self._zeros_like_span(b, i, j)
        return (self.entropy_c[b, j] - self.entropy_c[b, i]) / (j - i)

    def pitch_mean(self, b, i, j) -> torch.Tensor:
        """The span's mean pitch distribution ``(..., K)``.

        The mean of the softmaxes, not the softmax of mean logits: this is a *mixture*,
        which is what "the span's pitch content" means, and a mixture preserves the two
        modes of a span straddling a pitch change. A geometric mean would suppress
        exactly the structure the spread feature exists to detect."""
        if self.pitch_c is None:
            raise ValueError("no pitch evidence: this Evidence carries no pitch head")
        d = (j - i) if isinstance(j - i, int) else (j - i).unsqueeze(-1)
        return (self.pitch_c[b, j] - self.pitch_c[b, i]) / d

    def pitch_spread(self, b, i, j) -> torch.Tensor:
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


def pitch_jumps(contour: Contour, frame_times: np.ndarray) -> np.ndarray:
    """Per-frame indicator of a pyin pitch-change landing at each grid frame.

    A ``note`` span crossing a change is penalised, so the decoder can split a legato
    run at the pitch boundary even with no rest and a weak onset. ``group()`` once
    split on this cue too and no longer does -- pyin's transition lags a real onset,
    which made it place notes late enough to fall outside the match tolerance. Here it
    is a weighted *feature* rather than a hard boundary, so the same lag is something
    the decoder can learn to discount; that is a weaker claim than "this cue is good",
    and it is why the threshold lives here rather than being imported.

    Detection bridges pyin dropouts the way ``group()``'s running pitch does: a bass
    note transition usually has a one-frame unvoiced attack right at the boundary, so
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
        if not np.isnan(last) and abs(m - last) > JUMP_SEMITONES:
            jump[k] = 1.0
        last = m
    return jump


def voiced_logcumsum(
    voicing: np.ndarray | torch.Tensor | None,
    contour: Contour,
    frame_times: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix cumsums of ``(log P(voiced), log P(unvoiced))`` on the strided grid.

    Prefer the voicing head; with none, fall back to pyin's own voiced mask
    (``freqs > 0``) resampled to the frame grid, as a near-hard 0/1 probability -- the
    CRF still runs on DSP-only cues, just without a learned rest signal. Both branches
    land on ``device``: the fallback is numpy-born and would otherwise pin the cumsums
    to CPU while a head-fed run put them on the accelerator."""
    if voicing is not None:
        v = _on(voicing, device).float().clamp(1e-6, 1 - 1e-6)
        vlp, ulp = torch.log(v), torch.log1p(-v)
    else:
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
) -> tuple:
    """One track's per-frame tensors, pitch cumsums included when it has a pitch head."""
    frame_times = np.ascontiguousarray(cues.frame_times[::stride])
    # Stride first, place second: the slice is cheap on the source device and only the
    # decimated grid crosses the bus.
    onset = _on(torch.as_tensor(cues.onset).float()[::stride], device)
    onset_logit = torch.logit(onset.clamp(1e-6, 1 - 1e-6))
    voicing = cues.voicing[::stride] if cues.voicing is not None else None
    Vc, Uc = voiced_logcumsum(voicing, cues.contour, frame_times, device)
    # `jump` reads the pyin contour, which is numpy by TensorCues' invariant, so this
    # is the field that would otherwise strand the DP across two devices.
    jump = _on(pitch_jumps(cues.contour, frame_times), device)
    Jc = torch.cat([jump.new_zeros(1), torch.cumsum(jump, 0)])
    logits = getattr(cues, "pitch_logits", None)
    pitch = None if logits is None else pitch_cumsums(_on(logits, device)[::stride])
    return onset_logit, Vc, Uc, Jc, frame_times, pitch


def build_evidence(
    cues: Cues | TensorCues | list[Cues | TensorCues],
    stride: int,
    device: torch.device | str | None = None,
) -> Evidence:
    """Build the strided ``Evidence`` the DP scans, from one track or a list.

    ``device`` is the single placement every field is built onto, establishing the
    one-device invariant. Callers that own the scorer pass *its* module's device, so
    the numpy-born ``jump_c`` follows the parameters it will be added to -- which is
    what makes a numpy ``Cues`` decodable by a module on an accelerator.

    Padding is grad-preserving: the per-track tensors are scattered into buffers
    allocated off the first track."""
    batch = cues if isinstance(cues, list) else [cues]
    if not batch:
        raise ValueError("build_evidence needs at least one track")
    device = _cues_device(batch[0]) if device is None else torch.device(device)

    built = [_track_tensors(c, stride, device) for c in batch]
    lengths = [int(t[0].shape[0]) for t in built]
    B, Tmax = len(built), max(lengths)

    onset = built[0][0].new_zeros(B, Tmax)
    Vc = onset.new_zeros(B, Tmax + 1)
    Uc = onset.new_zeros(B, Tmax + 1)
    Jc = onset.new_zeros(B, Tmax + 1)
    has_pitch = built[0][5] is not None
    if has_pitch and any(t[5] is None for t in built):
        raise ValueError(
            "batch mixes tracks with and without pitch logits; every track in a batch "
            "must come from the same model, or the span features silently read zeros "
            "for some rows and real evidence for others"
        )
    K = built[0][5][2].shape[-1] if has_pitch else 0
    Pc = onset.new_zeros(B, Tmax + 1) if has_pitch else None
    Hc = onset.new_zeros(B, Tmax + 1) if has_pitch else None
    Dc = onset.new_zeros(B, Tmax + 1, K) if has_pitch else None

    for b, (o, v, u, jc, _, pitch) in enumerate(built):
        t = lengths[b]
        onset[b, :t] = o
        Vc[b, : t + 1], Vc[b, t + 1 :] = v, v[-1]  # flat past the end
        Uc[b, : t + 1], Uc[b, t + 1 :] = u, u[-1]
        Jc[b, : t + 1], Jc[b, t + 1 :] = jc, jc[-1]
        if pitch is not None:
            pc, hc, dc = pitch
            Pc[b, : t + 1], Pc[b, t + 1 :] = pc, pc[-1]
            Hc[b, : t + 1], Hc[b, t + 1 :] = hc, hc[-1]
            Dc[b, : t + 1], Dc[b, t + 1 :] = dc, dc[-1]

    return Evidence(
        onset, Vc, Uc, Jc, lengths,
        [t[4] for t in built], [c.contour for c in batch], Pc, Hc, Dc,
    )
