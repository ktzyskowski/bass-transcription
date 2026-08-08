"""The segmental recurrence, and its logsumexp reduction: ``log Z``.

``alpha[j][label]`` is the total mass of all ways to tile ``[0, j)`` ending in a
``label`` span, so ``log Z`` is the logsumexp over every segmentation. Differentiable
-- ``d log Z`` gives the model's expected segment features -- which is what makes it
the normaliser half of the NLL.

``step_terms`` is **the** recurrence, and the only place it is written. ``viterbi``
consumes the same tensor under a different reduction. The two reductions stay apart
because they differ in cost, not in logic: Viterbi needs a backpointer table and reads
an argmax back to the host every step, and making the forward pass go through a shared
reducer would have it pay for an argmax it discards and lose the static shapes that
let the scan be captured as one CUDA graph.

Two entry points over the one ``scan``:

  - ``log_partitions`` -- a padded batch of *ragged* whole tracks, each read out at its
    own length. A single track is a batch of one.
  - ``chunk_log_partition`` -- one fixed-size chunk, resuming from a carried alpha
    window. The truncated-BPTT entry point.

Every index tensor is built on the evidence's device (``Evidence`` guarantees there is
exactly one). Left on the default it would be CPU, which still *works* against CUDA
operands -- indexing silently transfers -- but pays a host-to-device copy per step of a
scan that runs the length of the track. See ``docs/semi-crf.md`` and
``docs/gpu-throughput-plan.md``.
"""

from typing import cast

import torch

from stemify.semicrf.evidence import Evidence
from stemify.semicrf.semiring import LOG, Semiring
from stemify.semicrf.spans import Grid, SpanScorer
from stemify.semicrf.window import initial_window, push


def candidate_starts(grid: Grid, j: int, device: torch.device) -> torch.Tensor:
    """``(dmax,)`` candidate start frames for a span ending at ``j``, oldest first.

    Clamped at 0 so the ramp steps never index out of bounds; those positions pair with
    a ``NEG`` alpha and so cannot win. Shared with ``viterbi``, which needs the same
    mapping to turn a backpointer's window offset back into a frame index -- deriving
    it twice is how an off-by-one becomes a decode that is wrong only near track
    starts."""
    return (torch.arange(-grid.dmax, 0, device=device) + j).clamp(min=0)


def step_terms(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    window: torch.Tensor,
    j: int,
    semiring: Semiring = LOG,
    witness: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(B, dmax, n_labels)``: the score of ending a span at ``j`` from each candidate
    start, with the previous label already summed (or maxed) out.

    This is **the** recurrence. Reducing the result over ``dim=1`` finishes the step:
    ``logsumexp`` gives the forward pass, ``max`` gives Viterbi.

    **The previous-label axis is contracted here rather than returned.** Written as one
    broadcast, ``alpha[i][prev] + trans[prev][next] + unary[next, i, j]`` is
    ``(B, dmax, n_rows, n_labels)`` -- fine at two labels, 9 MB per step at forty-nine
    and ~1.15 GB retained across a chunk. ``semiring.contract`` does the same arithmetic
    without materialising it, because ⊗ distributes over ⊕: the shared ``unary`` factors
    out of the sum over ``prev``, which is the identity the whole dynamic program rests
    on, applied once more.

    ``witness`` asks for the argmax over ``prev`` alongside, which Viterbi needs and the
    forward pass must not pay for. Returned as ``(term, arg_prev)``; ``arg_prev`` is
    ``None`` under a semiring with no survivor to point at.

    The window is read at its **full** ``dmax`` width rather than trimmed for
    ``j < dmax``. Out-of-range positions hold ``NEG``, so they contribute nothing:
    identical arithmetic, one *static* shape at every step. That is what makes the loop
    capturable as a single CUDA graph, and what lets a chunk resume from a carry.

    Candidate starts are clamped at 0 so the ramp steps never index out of bounds. The
    potentials they produce are meaningless, but they are *finite*, and they pair with a
    ``NEG`` alpha, so they cannot perturb the result.
    """
    B = window.shape[0]
    device = ev.device
    starts = candidate_starts(grid, j, device)                              # (dmax,)
    rows = torch.arange(B, device=device).unsqueeze(1)                      # (B, 1)
    unary = scorer.unary(ev, grid, rows, starts.unsqueeze(0), j)     # (B, dmax, n_labels)
    trans = scorer.transitions()                                    # (n_rows, n_labels)
    if not witness:
        return semiring.contract(window, trans) + unary, None
    survivor = semiring.contract_with_witness(window, trans)
    if survivor is None:
        raise ValueError(
            f"{type(semiring).__name__} combines the previous-label alternatives "
            "without keeping one, so it has no backpointer to record"
        )
    inner, arg_prev = survivor
    return inner + unary, arg_prev


def scan(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    window: torch.Tensor,
    ends: dict[int, list[int]] | None = None,
    captured: list[torch.Tensor | None] | None = None,
    start: int = 0,
    stop: int | None = None,
    *,
    semiring: Semiring = LOG,
    bias: torch.Tensor | None = None,
    perturb: torch.Tensor | None = None,
    witness: torch.Tensor | None = None,
) -> torch.Tensor:
    """The segmental recurrence from ``window``; returns the final alpha window.

    **The only loop over the recurrence.** ``semiring`` decides what the alternatives
    at each step combine into -- ``LOG`` sums them into ``log Z``, ``MAX_PLUS`` keeps
    the best -- and everything else here (candidate enumeration, the window slide, the
    ragged readout, the context trimming) is identical either way. Viterbi used to carry
    a second copy of this loop for the sake of one differing line.

    ``ends`` maps a step index to the batch rows whose track truly ends there; those
    rows' alpha are written into ``captured``. Only the whole-track path needs it -- a
    uniform chunk reads its answer off the returned window.

    ``start`` skips that many leading frames: they are *left context*, present only so
    spans reaching back past a chunk boundary can read their cue values, and already
    accounted for by the incoming ``window``. ``stop`` ends the scan early -- its mirror
    image, trailing frames that exist so the *encoder's* receptive field does not see
    zero-padding at the chunk's right edge, and which the DP must not consume.

    ``bias`` is added to every step's terms before the reduction, broadcast against
    ``(B, dmax, 3, 2)`` -- one constant for the whole scan.

    ``perturb`` is its per-span twin, ``(B, stop+1, dmax, 2)``, sliced at each step. It
    exists to be differentiated rather than to change an answer: with it left at zero
    the scan returns exactly what it otherwise would, and ``d log Z / d perturb`` is the
    posterior probability of each span. The two are separate parameters because they are
    separate ideas -- one is a correction someone tuned, the other is a handle autograd
    reads a marginal off. Collapsing them into one shape-polymorphic argument would save
    a line and lose that.

    ``witness`` is a ``(stop+1, B, 2)`` int64 table filled with the
    index of each step's surviving alternative -- backpointers, for a caller that means
    to reconstruct the path. Requesting one from a semiring that has no survivor to
    point at is an error rather than a silent no-op, because the traceback that follows
    would read zeros and return a plausible wrong segmentation.
    """
    if (ends is None) != (captured is None):
        raise ValueError(
            "`ends` and `captured` are two halves of the ragged whole-track readout -- "
            "where each row ends, and where its alpha is written -- so passing one "
            "without the other asks for a capture with nowhere to go"
        )
    B = ev.B
    stop = ev.T if stop is None else stop
    n_rows = scorer.labels.n_rows
    neg_col = ev.onset_logit.new_full((B, 1), semiring.zero)
    for j in range(start + 1, stop + 1):
        term, arg_prev = step_terms(
            scorer, ev, grid, window, j, semiring, witness=witness is not None
        )
        if bias is not None:
            term = term + bias
        if perturb is not None:
            # (B, dmax, n_labels): one epsilon per (start, label). The `prev` axis this
            # used to broadcast over is contracted away inside `step_terms` now, which
            # says the same thing it always did -- a span's identity does not include
            # what preceded it.
            term = term + perturb[:, j]
        if witness is None:
            val = semiring.reduce(term, dim=1)                 # (B, n_labels)
        else:
            survivor = semiring.reduce_with_witness(term, dim=1)
            if survivor is None:
                raise ValueError(
                    f"{type(semiring).__name__} combines alternatives without keeping "
                    "one, so it has no backpointer to record; a traceback over this "
                    "semiring would read an all-zero table and return a wrong "
                    "segmentation that looks like a right one"
                )
            val, arg_start = survivor                          # both (B, n_labels)
            assert arg_prev is not None                        # `witness` implies it
            # One packed backpointer per label, `offset * n_rows + prev` -- exactly the
            # layout the single flat argmax produced before the reduction was split in
            # two. The winning `prev` is the one belonging to the winning *start*, so it
            # is gathered at `arg_start`; reducing `arg_prev` again here would pick a
            # `prev` from some other start and return a path that was never scored.
            best_prev = arg_prev.gather(1, arg_start.unsqueeze(1)).squeeze(1)
            witness[j] = arg_start * n_rows + best_prev
        new = torch.cat([val, neg_col], dim=1)                 # (B, n_rows)
        if ends is not None and captured is not None:
            for b in ends.get(j, ()):
                captured[b] = new[b]
        window = push(window, new)
    return window


def log_partitions(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    perturb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log Z`` per track ``(B,)`` for a padded batch of ragged whole tracks.

    ``perturb`` rides through to ``scan``; see ``marginals`` for what differentiating
    with respect to it buys."""
    window = initial_window(ev.onset_logit, ev.B, grid.dmax, scorer.labels.n_rows)

    # Each track is read out at its *own* length, which the rolling window cannot hold:
    # lengths differ by far more than dmax (16617 vs 23355 on a real batch). So capture
    # alpha[L][b] as j passes each track's L, instead of retaining all Tmax.
    ends: dict[int, list[int]] = {}
    for b, length in enumerate(ev.lengths):
        ends.setdefault(length, []).append(b)
    captured: list[torch.Tensor | None] = [None] * ev.B
    for b in ends.get(0, ()):                          # an empty track ends at START
        captured[b] = window[b, -1]

    scan(scorer, ev, grid, window, ends, captured, perturb=perturb)
    # Every row is written: `ends` is keyed by `ev.lengths`, no length exceeds `ev.T`
    # (Tmax *is* the max), and the scan visits every j in [1, T]. Checked rather than
    # cast, because a row left unwritten means a track whose length the evidence and
    # the scan disagree about -- and `torch.stack` would report that as a bare TypeError
    # about NoneType, several frames from the cause.
    missing = [b for b, a in enumerate(captured) if a is None]
    if missing:
        raise RuntimeError(
            f"rows {missing} were never captured: their lengths "
            f"{[ev.lengths[b] for b in missing]} lie outside the scanned range "
            f"[1, {ev.T}]"
        )
    alphas = cast(list[torch.Tensor], captured)
    return torch.logsumexp(torch.stack(alphas)[:, : scorer.labels.n_labels], dim=1)


def chunk_log_partition(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    carry: torch.Tensor | None = None,
    context: int = 0,
    right: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One uniform-length chunk of the recurrence, resuming from ``carry``.

    Returns ``(A, window)``. ``A`` is the log-mass over every tiling of *everything
    scanned so far* -- this chunk and every chunk carried into it -- and ``window`` is
    the alpha state the next chunk resumes from. Pass ``carry=None`` for a track's
    first chunk.

    **``ev`` must carry ``context`` frames of left overlap ahead of the chunk's own
    frames, and ``context`` must be ``dmax`` unless the chunk starts at frame 0.** The
    carry supplies the *alpha* of positions before the chunk, but a span crossing the
    boundary also needs its *cue values*, and those live in the previous chunk. Without
    the overlap the scan silently clamps those starts to the chunk's own frame 0 and
    scores them against the wrong cues -- wrong rather than merely approximate.

    **``right`` is the mirror of ``context``: trailing frames the DP must not scan.**
    Scanning them would return a carry ``right`` frames past where the next chunk
    resumes, and those frames would then be scanned twice -- silently, since every
    shape still matches and the loss still falls.

    ``A`` is cumulative rather than per-chunk because that makes the training loss
    telescope. With ``A_0 = 0`` and the carry detached between chunks::

        sum_k [ (A_k - A_{k-1}) - S_k(gold) ]  ==  log Z - S(gold)

    exactly, while each term's gradient reaches only chunk ``k``'s activations.
    Detaching is what makes it truncated; passing the window through undetached
    reproduces whole-track backprop.
    """
    Tmax = ev.T
    if any(length != Tmax for length in ev.lengths):
        raise ValueError(
            f"chunk_log_partition needs uniform lengths, got {sorted(set(ev.lengths))} "
            f"against Tmax={Tmax}; use log_partitions for ragged whole tracks"
        )
    if not 0 <= context < Tmax:
        raise ValueError(f"context must be in [0, Tmax={Tmax}), got {context}")
    if not 0 <= right < Tmax - context:
        raise ValueError(
            f"right must be in [0, Tmax-context={Tmax - context}), got {right}; "
            "the chunk needs at least one frame of its own to scan"
        )
    window = (
        initial_window(ev.onset_logit, ev.B, grid.dmax, scorer.labels.n_rows)
        if carry is None else carry
    )
    window = scan(scorer, ev, grid, window, start=context, stop=Tmax - right)
    return torch.logsumexp(window[:, -1, : scorer.labels.n_labels], dim=1), window
