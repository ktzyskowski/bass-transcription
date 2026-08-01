"""The segmental forward recurrence: ``log Z``, in the logsumexp semiring.

``alpha[j][label]`` is the total mass of all ways to tile ``[0, j)`` ending in a
``label`` span, so ``log Z`` is the logsumexp over every segmentation. Differentiable
-- ``d log Z`` gives the model's expected segment features -- which is what makes it
the normaliser half of the NLL. ``viterbi`` runs the same recurrence in the max
semiring; ``score`` evaluates one given segmentation.

Three entry points over the one ``_scan``:

  - ``log_partition`` -- a single track, unbatched. The reference implementation the
    batched forms are tested against.
  - ``batched_log_partition`` -- a padded minibatch of *ragged* whole tracks, read out
    at each track's own length.
  - ``chunk_log_partition`` -- one fixed-size chunk of a track, resuming from a carried
    alpha window. The truncated-BPTT entry point.

Free functions over ``(potentials, emissions)`` -- they never touch a raw parameter,
so each is testable against a hand-set ``SemiCRFPotentials``. See ``docs/semi-crf.md``
and ``docs/gpu-throughput-plan.md``.

Every index tensor is built on the emissions' device (``Emissions`` guarantees there
is exactly one). Left on the default it would be CPU, which still *works* against
CUDA operands -- indexing silently transfers -- but pays a host-to-device copy per
step of a scan that runs the length of the track. Callers must hand in potentials and
emissions that agree; ``SemiCRFDecoder`` builds the emissions on the module's device
so that holds by construction.
"""

import torch

from stemify.decoders.semicrf.constants import NEG, START
from stemify.decoders.semicrf.emissions import BatchEmissions, Emissions
from stemify.decoders.semicrf.potentials import SemiCRFPotentials
from stemify.decoders.semicrf.window import initial_window, new_window, push


def log_partition(
    pot: SemiCRFPotentials, e: Emissions, dmax: int, dur_bucket: torch.Tensor
) -> torch.Tensor:
    """``log Z`` for a single track, unbatched.

    Trims the window to the reachable width each step rather than reading it full, so
    it stays the straightforward transcription of the recurrence -- the reference the
    batched forms are checked against."""
    T = e.T
    trans_to = pot.trans_to()  # (3, 2)
    a0 = e.onset_logit.new_full((3,), NEG)
    a0[START] = 0.0
    window, last = new_window(a0, dmax), a0
    for j in range(1, T + 1):
        lo = max(0, j - dmax)
        idx = torch.arange(lo, j, device=e.device)
        note = pot.note_potentials(e, idx, j, dur_bucket)
        rest = pot.rest_potentials(e, idx, j)
        seg = torch.stack([note, rest], dim=-1)          # (W, 2=next)
        prev = window[dmax - (j - lo):]                  # (W, 3=prev) -- a view
        term = prev.unsqueeze(-1) + trans_to.unsqueeze(0) + seg.unsqueeze(1)  # (W,3,2)
        val = torch.logsumexp(term.reshape(-1, 2), dim=0)  # (2,)
        last = torch.cat([val, val.new_full((1,), NEG)])
        window = push(window, last)
    return torch.logsumexp(last[:2], dim=0)


def _scan(
    pot: SemiCRFPotentials,
    batch: BatchEmissions,
    dmax: int,
    dur_bucket: torch.Tensor,
    window: torch.Tensor,
    ends: dict[int, list[int]] | None = None,
    captured: list[torch.Tensor | None] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    """The segmental forward recurrence over a padded minibatch, from ``window``.
    Returns the final alpha window.

    The window is read at its **full** ``dmax`` width at every step rather than
    trimmed for ``j < dmax``. Out-of-range positions hold ``NEG``, so ``exp(NEG - max)``
    underflows to exactly 0 in float32 and they contribute nothing to the logsumexp:
    identical arithmetic, but one *static* shape at every step. That is what makes the
    loop capturable as a single CUDA graph, and it is what lets a chunk resume from a
    previous chunk's carry -- with a carry there is no ramp at all, because every
    candidate start is a real frame in the chunk before.

    Candidate starts are clamped at 0 so the ramp steps never index out of bounds. The
    potentials they produce are meaningless, but they are *finite*, and they are added
    to a ``NEG`` alpha, so they cannot perturb the result.

    ``ends`` maps a step index to the batch rows whose track truly ends there; those
    rows' alpha are written into ``captured``. Only the whole-track path needs it -- a
    uniform chunk reads its answer off the returned window.

    ``start`` skips that many leading frames: they are *left context*, present only so
    spans reaching back past the chunk boundary can read their cue values, and already
    accounted for by the incoming ``window``.

    ``stop`` ends the scan early, defaulting to the full width. Its mirror image:
    trailing frames that exist only so the *encoder's* receptive field does not see
    zero-padding at the chunk's right edge, and which the DP must not consume -- the
    returned window has to land on the chunk boundary the next chunk resumes from."""
    onset, Vc, Uc, Jc = batch.onset_logit, batch.Vc, batch.Uc, batch.Jc
    B, Tmax = onset.shape
    stop = Tmax if stop is None else stop
    trans_to = pot.trans_to()                        # (3, 2)
    neg_col = onset.new_full((B, 1), NEG)
    offsets = torch.arange(-dmax, 0, device=onset.device)

    for j in range(start + 1, stop + 1):
        idx = (offsets + j).clamp(min=0)             # (dmax,) candidate starts
        d = j - idx
        vsum = Vc[:, j : j + 1] - Vc[:, idx]         # (B, dmax)
        usum = Uc[:, j : j + 1] - Uc[:, idx]
        jmp = Jc[:, j : j + 1] - Jc[:, idx + 1]      # interior pitch changes
        dur_term = pot.dur[dur_bucket[d]]            # (dmax,)
        note = (pot.w_on * onset[:, idx] + pot.w_v * vsum + pot.w_pitch * jmp
                + pot.b_note + dur_term.unsqueeze(0))
        rest = pot.w_u * usum + pot.b_rest
        seg = torch.stack([note, rest], dim=-1)      # (B, dmax, 2=next)
        term = window.unsqueeze(-1) + trans_to.view(1, 1, 3, 2) + seg.unsqueeze(2)
        val = torch.logsumexp(term.reshape(B, -1, 2), dim=1)  # (B, 2)
        new = torch.cat([val, neg_col], dim=1)                # (B, 3)
        if ends is not None:
            for b in ends.get(j, ()):
                captured[b] = new[b]
        window = push(window, new, dim=1)
    return window


def batched_log_partition(
    pot: SemiCRFPotentials, batch: BatchEmissions, dmax: int, dur_bucket: torch.Tensor
) -> torch.Tensor:
    """``log Z`` for every track in a padded minibatch: the same recurrence as
    ``log_partition`` run over ``(B, ...)`` tensors in one ``j``-loop, read out at
    each track's true length. Equal to the per-track scan track-for-track; this is
    only the throughput form (one kernel per step, not per track)."""
    B = batch.onset_logit.shape[0]
    window = initial_window(batch.onset_logit, B, dmax)

    # Each track is read out at its *own* length, which the rolling window cannot
    # hold: lengths differ by far more than dmax (16617 vs 23355 on a real batch).
    # So capture alpha[L][b] as j passes each track's L, instead of retaining all Tmax.
    ends: dict[int, list[int]] = {}
    for b, length in enumerate(batch.lengths):
        ends.setdefault(length, []).append(b)
    captured: list[torch.Tensor | None] = [None] * B
    for b in ends.get(0, ()):                         # an empty track ends at START
        captured[b] = window[b, -1]

    _scan(pot, batch, dmax, dur_bucket, window, ends, captured)
    final = torch.stack(captured)                    # (B, 3)
    return torch.logsumexp(final[:, :2], dim=1)      # (B,)


def chunk_log_partition(
    pot: SemiCRFPotentials,
    batch: BatchEmissions,
    dmax: int,
    dur_bucket: torch.Tensor,
    carry: torch.Tensor | None = None,
    context: int = 0,
    right: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One uniform-length chunk of the forward recurrence, resuming from ``carry``.

    Returns ``(A, window)``. ``A`` is the log-mass over every tiling of *everything
    scanned so far* -- this chunk and every chunk carried into it -- and ``window`` is
    the alpha state the next chunk resumes from. Pass ``carry=None`` for a track's
    first chunk.

    **``batch`` must carry ``context`` frames of left overlap ahead of the chunk's own
    frames, and ``context`` must be ``dmax`` unless the chunk starts at the track's
    frame 0.** The carry supplies the *alpha* of positions before the chunk, but a span
    crossing the boundary also needs its *cue values* -- ``onset[i]``, ``Vc[j]-Vc[i]``
    -- and those live in the previous chunk. Without the overlap the scan silently
    clamps those starts to the chunk's own frame 0 and scores them against the wrong
    cues, which is wrong rather than merely approximate. A chunk whose overlap reaches
    back to global frame 0 may pass a shorter ``context``: there the clamp is correct,
    and the carry holds ``NEG`` in the unreachable slots.

    **``right`` is the mirror of ``context``: trailing frames the DP must not scan.**
    They exist so the *encoder's* receptive field does not see zero-padding at the
    chunk's right edge, but they belong to the next chunk. Scanning them would return a
    carry ``right`` frames past where the next chunk resumes, and those frames would
    then be scanned twice -- silently, since every shape still matches and the loss
    still falls. Leaving it 0 (the default) is only correct when the window has no
    right margin.

    This is the truncated-BPTT entry point, and the reason ``A`` is cumulative rather
    than per-chunk is that it makes the training loss telescope. With ``A_0 = 0`` and
    the carry detached between chunks::

        sum_k [ (A_k - A_{k-1}) - S_k(gold) ]  ==  log Z - S(gold)

    exactly, while each term's gradient reaches only chunk ``k``'s encoder activations
    -- so a chunk can be backwarded and freed before the next one is built. Passing the
    previous ``window`` in *without* detaching instead reproduces whole-track backprop;
    detaching is what makes it truncated.

    Every track in the batch must have the same length: a chunk is read off the end of
    the window, not at per-track offsets. Use ``batched_log_partition`` for ragged
    whole tracks."""
    B, Tmax = batch.onset_logit.shape
    if any(length != Tmax for length in batch.lengths):
        raise ValueError(
            f"chunk_log_partition needs uniform lengths, got {sorted(set(batch.lengths))}"
            f" against Tmax={Tmax}; use batched_log_partition for ragged tracks"
        )
    if not 0 <= context < Tmax:
        raise ValueError(f"context must be in [0, Tmax={Tmax}), got {context}")
    if not 0 <= right < Tmax - context:
        raise ValueError(
            f"right must be in [0, Tmax-context={Tmax - context}), got {right}; "
            "the chunk needs at least one frame of its own to scan"
        )
    window = initial_window(batch.onset_logit, B, dmax) if carry is None else carry
    window = _scan(
        pot, batch, dmax, dur_bucket, window, start=context, stop=Tmax - right
    )
    return torch.logsumexp(window[:, -1, :2], dim=1), window
