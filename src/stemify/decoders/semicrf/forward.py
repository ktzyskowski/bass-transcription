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

import torch

from stemify.decoders.semicrf.constants import NEG
from stemify.decoders.semicrf.evidence import Evidence
from stemify.decoders.semicrf.spans import Grid, SpanScorer
from stemify.decoders.semicrf.window import initial_window, push


def candidate_starts(grid: Grid, j: int, device: torch.device) -> torch.Tensor:
    """``(dmax,)`` candidate start frames for a span ending at ``j``, oldest first.

    Clamped at 0 so the ramp steps never index out of bounds; those positions pair with
    a ``NEG`` alpha and so cannot win. Shared with ``viterbi``, which needs the same
    mapping to turn a backpointer's window offset back into a frame index -- deriving
    it twice is how an off-by-one becomes a decode that is wrong only near track
    starts."""
    return (torch.arange(-grid.dmax, 0, device=device) + j).clamp(min=0)


def step_terms(
    scorer: SpanScorer, ev: Evidence, grid: Grid, window: torch.Tensor, j: int
) -> torch.Tensor:
    """``(B, dmax, 3, 2)``: ``alpha[i][prev] + trans(prev -> next) + unary(next, i, j)``
    for every candidate start ``i`` and every label pair, at position ``j``.

    Reducing this over its middle two axes is the recurrence: ``logsumexp`` gives the
    forward pass, ``max`` gives Viterbi. Everything above the reduction is shared, and
    this is the only function that knows the recurrence's shape.

    The window is read at its **full** ``dmax`` width rather than trimmed for
    ``j < dmax``. Out-of-range positions hold ``NEG``, so ``exp(NEG - max)`` underflows
    to exactly 0 in float32 and they contribute nothing: identical arithmetic, one
    *static* shape at every step. That is what makes the loop capturable as a single
    CUDA graph, and what lets a chunk resume from a previous chunk's carry.

    Candidate starts are clamped at 0 so the ramp steps never index out of bounds. The
    potentials they produce are meaningless, but they are *finite*, and they are added
    to a ``NEG`` alpha, so they cannot perturb the result.
    """
    B = window.shape[0]
    device = ev.device
    starts = candidate_starts(grid, j, device)                              # (dmax,)
    rows = torch.arange(B, device=device).unsqueeze(1)                      # (B, 1)
    unary = scorer.unary(ev, grid, rows, starts.unsqueeze(0), j)            # (B, dmax, 2)
    return (
        window.unsqueeze(-1)                        # (B, dmax, 3=prev, 1)
        + scorer.transitions().view(1, 1, 3, 2)     # (1, 1,    3, 2=next)
        + unary.unsqueeze(2)                        # (B, dmax, 1, 2)
    )


def scan(
    scorer: SpanScorer,
    ev: Evidence,
    grid: Grid,
    window: torch.Tensor,
    ends: dict[int, list[int]] | None = None,
    captured: list[torch.Tensor | None] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> torch.Tensor:
    """The forward recurrence from ``window``; returns the final alpha window.

    ``ends`` maps a step index to the batch rows whose track truly ends there; those
    rows' alpha are written into ``captured``. Only the whole-track path needs it -- a
    uniform chunk reads its answer off the returned window.

    ``start`` skips that many leading frames: they are *left context*, present only so
    spans reaching back past a chunk boundary can read their cue values, and already
    accounted for by the incoming ``window``. ``stop`` ends the scan early -- its mirror
    image, trailing frames that exist so the *encoder's* receptive field does not see
    zero-padding at the chunk's right edge, and which the DP must not consume.
    """
    B = ev.B
    stop = ev.T if stop is None else stop
    neg_col = ev.onset_logit.new_full((B, 1), NEG)
    for j in range(start + 1, stop + 1):
        term = step_terms(scorer, ev, grid, window, j)
        val = torch.logsumexp(term.reshape(B, -1, 2), dim=1)   # (B, 2)
        new = torch.cat([val, neg_col], dim=1)                 # (B, 3)
        if ends is not None:
            for b in ends.get(j, ()):
                captured[b] = new[b]
        window = push(window, new)
    return window


def log_partitions(scorer: SpanScorer, ev: Evidence, grid: Grid) -> torch.Tensor:
    """``log Z`` per track ``(B,)`` for a padded batch of ragged whole tracks."""
    window = initial_window(ev.onset_logit, ev.B, grid.dmax)

    # Each track is read out at its *own* length, which the rolling window cannot hold:
    # lengths differ by far more than dmax (16617 vs 23355 on a real batch). So capture
    # alpha[L][b] as j passes each track's L, instead of retaining all Tmax.
    ends: dict[int, list[int]] = {}
    for b, length in enumerate(ev.lengths):
        ends.setdefault(length, []).append(b)
    captured: list[torch.Tensor | None] = [None] * ev.B
    for b in ends.get(0, ()):                          # an empty track ends at START
        captured[b] = window[b, -1]

    scan(scorer, ev, grid, window, ends, captured)
    return torch.logsumexp(torch.stack(captured)[:, :2], dim=1)


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
    window = initial_window(ev.onset_logit, ev.B, grid.dmax) if carry is None else carry
    window = scan(scorer, ev, grid, window, start=context, stop=Tmax - right)
    return torch.logsumexp(window[:, -1, :2], dim=1), window
