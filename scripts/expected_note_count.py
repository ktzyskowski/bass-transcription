"""How many notes does the model *think* are there, versus how many Viterbi emits?

The bias sweep showed that one scalar recovers +6.7 separated F1 from an unchanged
checkpoint, and that F1 peaks where the emitted count crosses the reference count. Both
are consistent with the decode sitting in the wrong place on a granularity axis -- but
both were measured against labels, so neither separates "the readout is biased" from
"a count-shaped knob happens to help".

This separates them, without labels. For a log-linear model, the gradient of the log
partition function with respect to a feature's weight is that feature's **expected
value** under the model's own distribution:

    d log Z / d w_f = E_model[ f(y) ]

``b_note`` is the weight on a feature whose value is literally "number of note spans in
this segmentation". So one backward pass through the existing (differentiable) forward
recurrence gives ``E_model[note count]`` per track -- what the distribution believes,
with no reference notes involved and nothing tuned.

The prediction under the mode/mean diagnosis is sharp: **E_model[count] should sit near
the reference count while Viterbi's count sits well below it.** If instead the
expectation tracks Viterbi, the distribution genuinely believes in fewer notes, the
diagnosis is wrong, and marginal decoding cannot help -- the model would need retraining
rather than re-reading.

``--solve-bias`` then bisects for the per-track ``note_bias`` that brings Viterbi's
count up to the model's own expectation. That number is fitted to *no* validation set,
which is the point: if it lands near the +0.25/+0.5 the labelled sweep chose, the
correction is a property of the model rather than of these 37 tracks.
"""

import argparse
import statistics
from pathlib import Path

import torch

from stemify.core.cache import Cache
from stemify.semicrf.marginals import span_marginals
from stemify.semicrf.viterbi import viterbi
from stemify.models.learned_encoder import LearnedEncoder
from stemify.training.checkpoint import Checkpoint
from stemify.training.validation import build_validators

ROOT = "data/raw/slakh2100"
CACHE_DIR = "data/cache"


def expected_notes(decoder, cues) -> tuple[float, int]:
    """``(E_model[note spans], viterbi note spans)`` for one track.

    The expectation comes from ``autograd``, not a hand-written backward recursion:
    reverse mode through the forward scan *is* the backward algorithm, so the quantity
    is already being computed on every training step and only needs reading out.

    Summed out of the span marginals rather than differentiated against a *named*
    parameter. It used to read ``d log Z / d b_note``, which is the same number only
    while a scorer happens to carry a scalar that adds one to every note span --
    ``LinearSpanScorer`` does, a neural one does not. By linearity of expectation
    ``E[count] = sum over spans of P(span)``, so summing the marginals over the pitch
    labels asks the distribution directly and works for any scorer.
    """
    ev, grid = decoder._prepare([cues])
    labels = decoder.scorer.labels
    marginal = span_marginals(decoder.scorer, ev, grid)   # (1, T+1, dmax, n_labels)
    expected = float(marginal[..., : labels.n_pitches].sum())
    segs = viterbi(decoder.scorer, ev, grid)
    return expected, sum(1 for label, _, _ in segs if labels.is_note(label))


def solve_bias(decoder, cues, target: float, lo: float = -2.0, hi: float = 3.0,
               iters: int = 9) -> float:
    """The ``note_bias`` at which Viterbi emits ``target`` note spans.

    Bisection is valid because the count is monotone in the bias -- ``score(y) + b*k(y)``
    has increasing differences in ``(b, k)``, pinned by a test in ``test_semicrf``.
    """
    ev, grid = decoder._prepare([cues])
    labels = decoder.scorer.labels
    count = lambda b: sum(
        1 for label, _, _ in viterbi(decoder.scorer, ev, grid, note_bias=b)
        if labels.is_note(label)
    )
    for _ in range(iters):
        mid = (lo + hi) / 2
        if count(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("data/cache/semicrf_pitch_aux.pt"))
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--conditions", default="clean,separated")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--solve-bias", action="store_true",
                   help="also bisect the per-track bias that hits the expectation "
                        "(one Viterbi per iteration, so ~10x the decode cost)")
    args = p.parse_args()

    ck = Checkpoint.load(args.checkpoint)
    model = ck.recipe.build_model()
    model.load_state_dict(ck.model_state)
    decoder = ck.recipe.decoder_from({"loss": ck.objective_state})
    decoder.to(torch.device("cpu"))
    encoder = LearnedEncoder(model, device="cuda" if torch.cuda.is_available() else "cpu")

    validators = build_validators(
        args.root, args.limit, args.conditions.split(","), cache=Cache(args.cache_dir)
    )
    for tag, validator in validators.items():
        print(f"\n=== {tag} ===")
        print(f"{'track':>6} {'gold':>7} {'E[count]':>9} {'viterbi':>8} "
              f"{'E/gold':>7} {'vit/gold':>9}" + ("  bias" if args.solve_bias else ""))
        rows = []
        for i, ex in enumerate(validator.examples):
            cues = encoder.encode_features(ex.contour, ex.load_features())
            exp, vit = expected_notes(decoder, cues)
            gold = len(ex.notes)
            line = (f"{i:>6} {gold:>7} {exp:>9.1f} {vit:>8} "
                    f"{exp / max(gold, 1):>7.2f} {vit / max(gold, 1):>9.2f}")
            bias = None
            if args.solve_bias:
                bias = solve_bias(decoder, cues, exp)
                line += f"  {bias:+.2f}"
            print(line)
            rows.append((gold, exp, vit, bias))

        gold = sum(r[0] for r in rows)
        exp = sum(r[1] for r in rows)
        vit = sum(r[2] for r in rows)
        print(f"{'TOTAL':>6} {gold:>7} {exp:>9.1f} {vit:>8.0f} "
              f"{exp / gold:>7.2f} {vit / gold:>9.2f}")
        if args.solve_bias:
            biases = [r[3] for r in rows if r[3] is not None]
            print(f"  per-track bias: median {statistics.median(biases):+.2f}  "
                  f"mean {statistics.fmean(biases):+.2f}  "
                  f"range {min(biases):+.2f}..{max(biases):+.2f}")


if __name__ == "__main__":
    main()
