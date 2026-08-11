"""Is there anything left in MBR decoding, or is MAP-plus-bias the end of the line?

Four decision rules that decoded from span posteriors have failed, and they failed for
one shared reason: a DP can only maximise a quantity that **decomposes additively over
spans**, so each was a guess at which decomposable proxy stands in for the note metric
-- exact spans, exact boundaries, boundaries within the matcher's tolerance. Sampled MBR
has no such guess in it. Draw whole segmentations, score candidates against the *real*
matcher, keep the one with the highest expected agreement with the others.

There is a positive argument too. The bias this whole line of work exists to correct is
a property of ``max``: ``log Z`` accounts for the combinatorially many ways to place
``k`` boundaries, and Viterbi pays that per-span cost without seeing the multiplicity.
Sampling does not maximise anything, so it inherits the distribution's own expected note
count -- ~0.97 of the reference against MAP's 0.38 -- for free, where every deterministic
rule had to re-earn it through a proxy.

**This script does not build MBR. It bounds it.** For each track it draws ``K``
segmentations by forward-filtering backward-sampling and reports four numbers:

  * ``MAP+bias``  -- the incumbent, the same decode the sweep tunes.
  * ``sample``    -- the mean F1 of a single random draw. What the pool is worth on
                     average, and a floor: MBR that picks arbitrarily lands here.
  * ``MBR``       -- the consensus pick, chosen by pairwise F1 among the samples with
                     **no labels**. The real rule, approximated by a finite pool.
  * ``ORACLE``    -- the best sample per track, chosen *with* labels. The ceiling of
                     any selection rule over this pool, MBR included.

The verdict is read off the ordering, and the oracle is what makes it a verdict rather
than a fifth ambiguous null:

  * ``ORACLE <= MAP+bias`` -- **dead**, and provably so. No selection rule can beat a
    labelled oracle over the pool it selects from, so no cleverer consensus, no larger
    K, and no better utility function recovers it. The distribution does not contain a
    better segmentation than the one Viterbi already finds.
  * ``ORACLE >> MAP+bias`` and ``MBR`` near it -- build it. The pool holds the wins and
    consensus finds them without labels.
  * ``ORACLE >> MAP+bias`` but ``MBR`` near ``sample`` -- the wins are there and the
    unlabelled rule cannot find them. A bigger K or a different utility might; that is a
    real experiment rather than a guess.

Validation only. Everything here selects, and selection on test is how a held-out set
stops being one.
"""

import argparse
import time
from pathlib import Path

import torch

from stemify.persistence.store import Store
from stemify.semicrf.constants import NEG, NOTE, START
from stemify.semicrf.evidence import Evidence
from stemify.semicrf.forward import candidate_starts, log_partitions, step_terms
from stemify.semicrf.marginals import span_marginals
from stemify.semicrf.spans import SpanScorer
from stemify.semicrf.window import initial_window, push
from stemify.models.learned_encoder import LearnedEncoder
from stemify.metrics.note import count_matches, prf_rates
from stemify.music.notes import NoteEvent
from stemify.training.checkpoint import Checkpoint
from stemify.training.validation import build_validators

ROOT = "data/raw/slakh2100"
CACHE_DIR = "data/cache"
ENCODE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECODE_DEVICE = "cpu"

Segs = list[tuple[int, int, int]]


def forward_retaining(
    scorer: SpanScorer, ev: Evidence, dmax: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The forward scan, keeping every step's alpha *and* its term tensor.

    ``scan`` returns only the rolling window, which is all ``log Z`` needs and all a
    bounded-memory forward pass should keep. Backward sampling needs to revisit
    arbitrary positions, so this re-runs the same recurrence retaining what it discards.
    Deliberately a copy in a throwaway script rather than a flag on ``scan``: the
    library's scan is the one the training loop and the CUDA graph capture depend on,
    and retention is exactly the property that makes it capturable and O(dmax) in
    memory.

    Correctness is checked against ``log_partitions`` by the caller rather than argued
    -- the two must agree to float tolerance or this is scanning a different model.
    """
    window = initial_window(ev.onset_logit, 1, dmax, scorer.labels.n_rows)
    alphas = [window[:, -1, :]]                                        # alpha at j = 0
    terms: list[torch.Tensor] = [ev.onset_logit.new_zeros(1, dmax, 3, 2)]  # j=0 unused
    neg_col = ev.onset_logit.new_full((1, 1), NEG)
    for j in range(1, ev.T + 1):
        term, _ = step_terms(scorer, ev, dmax, window, j)          # (1, dmax, n_labels)
        terms.append(term)
        val = torch.logsumexp(term.reshape(1, -1, 2), dim=1)           # (1, 2)
        new = torch.cat([val, neg_col], dim=1)                         # (1, 3)
        alphas.append(new)
        window = push(window, new)
    return alphas, terms


def sample_segmentation(
    alphas: list[torch.Tensor],
    terms: list[torch.Tensor],
    dmax: int,
    T: int,
    device: torch.device,
    gen: torch.Generator,
) -> Segs:
    """One draw from ``P(y | x)`` by forward-filtering backward-sampling.

    Walks backwards from ``T``. Given that a span with label ``L`` ends at ``j``, the
    joint over (where it started, what preceded it) is exactly this step's term tensor
    with ``L`` fixed -- ``alpha[i][prev] + trans[prev][L] + unary(L, i, j)`` -- because
    the alphas already hold the total mass of every way to reach each ``i``. Normalising
    that and drawing from it is a draw from the true conditional, so the chain of draws
    is a draw from the full posterior. No rejection, no burn-in: it is exact.

    Positions before a track's start are clamped candidates paired with a ``log 0``
    alpha, so they normalise to exactly zero probability and cannot be drawn -- which is
    also why the walk terminates at frame 0 rather than needing a boundary case.
    """
    logits = alphas[T][0, :2]
    label = int(torch.multinomial(torch.softmax(logits, dim=0), 1, generator=gen))
    segs: Segs = []
    j = T
    while j > 0:
        weights = terms[j][0, :, :, label].reshape(-1)          # (dmax * 3,) over (d, prev)
        draw = int(torch.multinomial(torch.softmax(weights, dim=0), 1, generator=gen))
        d, prev = divmod(draw, 3)
        i = int(candidate_starts(dmax, j, device)[d])
        if prev == START and i != 0:
            raise RuntimeError(
                f"drew START as the predecessor of a span beginning at frame {i}, but "
                "START is only reachable at 0 -- the alpha window and the candidate "
                "starts have fallen out of step"
            )
        segs.append((label, i, j))
        j, label = i, prev
    segs.reverse()
    return segs


def check_tiling(segs: Segs, T: int, dmax: int) -> None:
    """A sample must be a segmentation, not merely a list of spans."""
    covered = [(i, j) for _, i, j in segs]
    if not covered or covered[0][0] != 0 or covered[-1][1] != T:
        raise RuntimeError(f"sample does not cover [0, {T}): {covered[:3]}...{covered[-3:]}")
    if any(a[1] != b[0] for a, b in zip(covered, covered[1:], strict=True)):
        raise RuntimeError("sample has a gap or an overlap")
    if any(not 1 <= j - i <= dmax for i, j in covered):
        raise RuntimeError(f"sample has a span outside [1, {dmax}]")


def f1(ref: list[NoteEvent], est: list[NoteEvent]) -> float:
    """Note F1 through the same matcher the headline metric uses.

    ``count_matches`` rather than ``score_notes`` because the consensus loop calls this
    O(K^2) times per track and has no use for fragmentation. Symmetric in its arguments
    -- swapping them swaps fp and fn, and F1 depends on them only through their sum --
    which is what lets the pairwise table be filled in half the calls.
    """
    return prf_rates(*count_matches(ref, est))["f1"]


def consensus(candidates: list[list[NoteEvent]]) -> int:
    """Index of the sample with the highest mean F1 against the others. **No labels.**

    This is MBR proper under an F1 utility: pick the candidate minimising expected loss
    where the expectation is taken over the model's own posterior, approximated by the
    sample pool standing in for it. The winner is the most *central* segmentation, not
    the highest-scoring one -- the model has no access to which is which.
    """
    k = len(candidates)
    total = [0.0] * k
    for a in range(k):
        for b in range(a + 1, k):
            score = f1(candidates[a], candidates[b])
            total[a] += score
            total[b] += score
    return max(range(k), key=lambda a: total[a])


def probe(
    checkpoint: Path, root: str, cache_dir: str, conditions: list[str],
    limit: int, samples: int, bias: float, seed: int,
) -> None:
    ck = Checkpoint.load(checkpoint)
    model = ck.recipe.build_model()
    model.load_state_dict(ck.model_state)
    decoder = ck.recipe.decoder_from({"loss": ck.objective_state})
    decoder.to(torch.device(DECODE_DEVICE))
    encoder = LearnedEncoder(model, device=ENCODE_DEVICE)
    print(f"{checkpoint}  pitch_head={encoder.has_pitch_head}  "
          f"encode={ENCODE_DEVICE} decode={DECODE_DEVICE}  K={samples}  bias={bias}")

    gen = torch.Generator(device=DECODE_DEVICE).manual_seed(seed)
    validators = build_validators(root, limit, conditions, cache=Store(cache_dir))
    for tag, validator in validators.items():
        examples = validator.examples
        t0 = time.perf_counter()
        cues = [encoder.encode_features(ex.contour, ex.load_features()) for ex in examples]
        gold = sum(len(ex.notes) for ex in examples)
        print(f"\n=== {tag}: {len(examples)} tracks, {gold} reference notes "
              f"(encoded in {time.perf_counter() - t0:.0f}s) ===")

        rows: dict[str, list[float]] = {k: [] for k in ("map", "sample", "mbr", "oracle")}
        counts: dict[str, list[float]] = {k: [] for k in rows}
        drift: list[float] = []
        for ex, cue in zip(examples, cues, strict=True):
            ref = ex.notes
            # `_prepare` and `_segs_to_notes` are the decoder's internals. Reached into
            # deliberately: the point of the probe is to compare decision rules over the
            # *identical* evidence and pitch readout, so re-deriving either here would
            # measure the re-derivation too.
            ev, dmax = decoder._prepare([cue])
            t0 = time.perf_counter()
            with torch.no_grad():
                alphas, terms = forward_retaining(decoder.scorer, ev, dmax)
                mine = float(torch.logsumexp(alphas[ev.T][0, :2], dim=0))
                theirs = float(log_partitions(decoder.scorer, ev, dmax)[0])
            if abs(mine - theirs) > 1e-2 * max(1.0, abs(theirs)):
                raise RuntimeError(
                    f"retained-alpha scan says log Z = {mine}, the library's scan says "
                    f"{theirs} -- this is not the same recurrence"
                )

            drawn = [
                sample_segmentation(alphas, terms, dmax, ev.T, ev.device, gen)
                for _ in range(samples)
            ]
            for segs in drawn:
                check_tiling(segs, ev.T, dmax)
            # FFBS is right only if the draws reproduce a quantity computed a completely
            # different way. The expected note count comes from the marginals, which are
            # verified against brute-force enumeration, so agreement here ties the
            # sampler to something already known to be correct.
            expected = float(span_marginals(decoder.scorer, ev, dmax)[0, :, :, NOTE].sum())
            drawn_mean = sum(sum(1 for lab, _, _ in s if lab == NOTE) for s in drawn) / samples
            drift.append(drawn_mean / expected if expected else float("nan"))

            cands = [decoder._segs_to_notes(s, ev) for s in drawn]
            scores = [f1(ref, c) for c in cands]
            best_by_consensus = consensus(cands)
            mapped = decoder.decode(cue, note_bias=bias)

            n = max(len(ref), 1)
            rows["map"].append(f1(ref, mapped))
            rows["sample"].append(sum(scores) / samples)
            rows["mbr"].append(scores[best_by_consensus])
            rows["oracle"].append(max(scores))
            counts["map"].append(len(mapped) / n)
            counts["sample"].append(sum(len(c) for c in cands) / samples / n)
            counts["mbr"].append(len(cands[best_by_consensus]) / n)
            counts["oracle"].append(len(cands[max(range(samples), key=lambda k: scores[k])]) / n)
            print(f"  {ex.track_id if hasattr(ex, 'track_id') else '?':<22} "
                  f"T={ev.T:<6d} map={rows['map'][-1]:.4f} "
                  f"sample={rows['sample'][-1]:.4f} mbr={rows['mbr'][-1]:.4f} "
                  f"oracle={rows['oracle'][-1]:.4f}  ({time.perf_counter() - t0:.0f}s)")
            del alphas, terms

        print(f"\n  sampled/expected note count {sum(drift) / len(drift):.3f} "
              f"(1.000 means the sampler agrees with the marginals)")
        print(f"  {'rule':<14} {'F1':>8} {'notes/gold':>11}")
        for key, name in (("map", f"MAP+bias {bias}"), ("sample", "sample mean"),
                          ("mbr", f"MBR (K={samples})"), ("oracle", "ORACLE best")):
            print(f"  {name:<14} {sum(rows[key]) / len(rows[key]):>8.4f} "
                  f"{sum(counts[key]) / len(counts[key]):>11.2f}")
        print("  F1 is macro-averaged over tracks -- the oracle is a per-track max, so "
              "a pooled micro F1 would not be the same quantity.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("data/cache/semicrf_pitch_aux2.epoch4.pt"))
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--conditions", default="separated")
    p.add_argument("--limit", type=int, default=8, help="val tracks per condition")
    p.add_argument("--samples", type=int, default=32,
                   help="draws per track; the consensus loop is O(K^2) in the matcher")
    p.add_argument("--bias", type=float, default=1.5,
                   help="the incumbent's tuned note bias, for the row it must beat")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    probe(args.checkpoint, args.root, args.cache_dir, args.conditions.split(","),
          args.limit, args.samples, args.bias, args.seed)


if __name__ == "__main__":
    main()
