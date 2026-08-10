"""Learned pitch head vs pyin, on one fixed checkpoint.

The cleanest A/B available: the model is frozen, both arms share the same onset and
voicing curves from the same forward pass, and the *only* difference is where the pitch
contour came from. Any change in note F1 is attributable to the pitch source alone.

Why it should help, and where: pyin runs at hop 512, twice the CQT's frame period, and
its pitch transition lags a real onset -- worse on separated audio, where it more often
never reaches the note's pitch at all. So the head should win by more on `separated`
than on `clean`, and if it loses on clean that is the trade to weigh rather than a
reason to discard it.

Pitch-blind F1 is reported beside strict F1 to separate *placement* from *pitch*: it
scores the same notes with the pitch tolerance relaxed away, so the difference between
the two columns is what pitch alone costs.

It is **not** a control that should come out equal between arms, which is what this
docstring used to claim. Switching the contour also switches the grid `group_contour`
segments on (23.2 ms to 11.6 ms) and changes how the voicing mask resamples, so
placement genuinely moves too. What the two arms compare is the learned pitch source as
a package -- accuracy plus grid resolution -- which is the right thing to measure but a
wider claim than "only the pitch changed".
"""

import argparse
from pathlib import Path

from stemify.persistence.store import Store
from stemify.core.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.training.checkpoint import load_cuenet
from stemify.core.metrics.note.base import matched_pairs
from stemify.core.metrics.note.prf import prf_rates
from stemify.core.conditions import CONDITIONS, canonical
from stemify.training.validation import CheckpointValidator

ROOT = "data/raw/slakh2100"
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--conditions", default="clean,separated")
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    model = load_cuenet(args.checkpoint)
    decoder = GreedyDecoder()
    cache = Store("data/cache")

    print(f"\n{args.checkpoint.stem}\n")
    print(f"{'condition':<12} {'split':<11} {'pitch':<9} {'recall':>8} {'prec':>7} "
          f"{'blind F1':>9} {'strict F1':>10} {'pitch gap':>10}")
    for tag in args.conditions.split(","):
        condition = CONDITIONS[canonical(tag)]
        for split in args.splits.split(","):
            v = CheckpointValidator(
                condition.dataset(ROOT, split, args.limit), condition, cache=cache
            )
            for use_pyin in (True, False):
                enc = LearnedEncoder(model, use_pyin=use_pyin)
                if not enc.has_pitch_head:
                    raise SystemExit(
                        f"{args.checkpoint} has no pitch head; there is no A/B to run."
                    )
                tp = fp = fn = stp = sfp = sfn = 0
                for ex in v.examples:
                    est = decoder.decode(
                        enc.encode_features(ex.contour, ex.load_features())
                    )
                    m = len(matched_pairs(ex.notes, est, pitch_tolerance=IGNORE_PITCH))
                    tp, fp, fn = tp + m, fp + len(est) - m, fn + len(ex.notes) - m
                    s = len(matched_pairs(ex.notes, est))
                    stp, sfp, sfn = stp + s, sfp + len(est) - s, sfn + len(ex.notes) - s
                b, st = prf_rates(tp, fp, fn), prf_rates(stp, sfp, sfn)
                print(f"{tag:<12} {split:<11} {enc.pitch_source:<9} {b['recall']:>8.4f} "
                      f"{b['precision']:>7.4f} {b['f1']:>9.4f} {st['f1']:>10.4f} "
                      f"{b['f1'] - st['f1']:>+10.4f}")


if __name__ == "__main__":
    main()
