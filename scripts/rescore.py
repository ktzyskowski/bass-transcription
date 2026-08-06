"""Re-score saved checkpoints under the current decoder.

Every note-level number in docs/experiments.md was produced before
``PITCH_SPLIT_SEMITONES`` was turned off, and that constant was worth ~7.5 strict F1 --
so those numbers do not compare to anything measured after it. Rather than re-train,
re-decode: the checkpoints are fixed, only the decode changed, so this recovers the
whole table for the cost of a forward pass per track.

It also settles bichannel on equal terms. ``recall-w5`` confounds the mix channel with
``pos_weight=5``; ``balanced`` is the clean 2-channel arm against ``rung0-mono``.
"""

import argparse
from pathlib import Path

from stemify.core.cache import Cache
from stemify.core.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.training.checkpoint import load_cuenet
from stemify.core.metrics.note.base import matched_pairs
from stemify.core.metrics.note.prf import prf_rates
from stemify.core.conditions import CONDITIONS
from stemify.training.validation import CheckpointValidator

ROOT = "data/raw/slakh2100"
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only

# (checkpoint, the condition it was trained for). Channel width is fixed by the
# checkpoint, so each arm has exactly one condition it can legally be scored on.
ARMS = [
    ("rung0-mono", "separated"),
    ("balanced", "bichannel-separated"),
    ("recall-w5", "bichannel-separated"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--splits", default="validation,test")
    args = p.parse_args()

    decoder = GreedyDecoder()
    print(f"{'arm':<12} {'condition':<21} {'split':<11} {'recall':>8} {'prec':>7} "
          f"{'blind F1':>9} {'strict F1':>10} {'pitch gap':>10}")
    for name, tag in ARMS:
        condition = CONDITIONS[tag]
        encoder = LearnedEncoder(load_cuenet(Path(f"data/checkpoints/{name}.pt")))
        for split in args.splits.split(","):
            v = CheckpointValidator(
                condition.dataset(ROOT, split, args.limit), condition, cache=Cache("data/cache")
            )
            tp = fp = fn = stp = sfp = sfn = 0
            for ex in v.examples:
                est = decoder.decode(encoder.encode_features(ex.contour, ex.load_features()))
                m = len(matched_pairs(ex.notes, est, pitch_tolerance=IGNORE_PITCH))
                tp, fp, fn = tp + m, fp + len(est) - m, fn + len(ex.notes) - m
                s = len(matched_pairs(ex.notes, est))
                stp, sfp, sfn = stp + s, sfp + len(est) - s, sfn + len(ex.notes) - s
            b, st = prf_rates(tp, fp, fn), prf_rates(stp, sfp, sfn)
            # The pitch gap is what a perfect pitch head would recover, holding the
            # segmentation fixed -- the lever step 2 is aimed at.
            print(f"{name:<12} {tag:<21} {split:<11} {b['recall']:>8.4f} "
                  f"{b['precision']:>7.4f} {b['f1']:>9.4f} {st['f1']:>10.4f} "
                  f"{b['f1'] - st['f1']:>+10.4f}")


if __name__ == "__main__":
    main()
