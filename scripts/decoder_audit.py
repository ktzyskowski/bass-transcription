"""Does each heuristic-decoder constant still earn its place?

Every constant in ``decoders/heuristic.py`` was chosen for a pipeline whose only cues
were pyin and spectral flux. Trained onset and voicing heads have since taken over the
jobs several of them were invented to do, and ``PITCH_SPLIT_SEMITONES`` showed what
happens when a superseded heuristic keeps competing: it cost ~7.5 strict F1 by
overriding a 7 ms-accurate onset with a ~58 ms-late pitch transition.

This audits the rest. The question per constant is *"can it be removed"*, not *"what is
its optimum"* -- deliberately. Six knobs tuned jointly on 40 validation tracks is a lot
of selection pressure on a small set, and the pitch-split result was trustworthy
precisely because it was monotone to the boundary and had a mechanism behind it, not
because it was an argmax. So each constant is swept alone, and a setting only wins if
it is both better *and* explainable.

Selection happens on validation. ``--split test`` re-runs a chosen configuration on
held-out data, which is the only number worth quoting.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from stemify.persistence.store import Store
from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.training.checkpoint import load_cuenet
from stemify.core.metrics.note.base import matched_pairs
from stemify.core.metrics.note.fragmentation import Fragmentation
from stemify.core.metrics.note.prf import prf_rates
from stemify.core.conditions import CONDITIONS, canonical
from stemify.training.validation import CheckpointValidator

ROOT = "data/raw/slakh2100"
CACHE_DIR = "data/cache"
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only
OFF = 1e9           # a pitch split this wide never fires -- the "removed" setting


@dataclass
class Scored:
    recall: float
    precision: float
    blind_f1: float
    strict_f1: float
    frag: int
    offpitch: int


def score(refs, ests) -> Scored:
    """Pitch-blind and strict counts from one shared match per track.

    Both altitudes come from the same estimates so the pitch gap between them is
    exactly the pitch error, holding segmentation fixed.
    """
    tp = fp = fn = stp = sfp = sfn = 0
    for ref, est in zip(refs, ests, strict=True):
        m = len(matched_pairs(ref, est, pitch_tolerance=IGNORE_PITCH))
        tp, fp, fn = tp + m, fp + len(est) - m, fn + len(ref) - m
        s = len(matched_pairs(ref, est))
        stp, sfp, sfn = stp + s, sfp + len(est) - s, sfn + len(ref) - s
    blind, strict = prf_rates(tp, fp, fn), prf_rates(stp, sfp, sfn)
    frag = Fragmentation().compute(refs, ests).scalars
    return Scored(
        blind["recall"], blind["precision"], blind["f1"], strict["f1"],
        int(frag["fragmentation"]), int(frag["interior_offpitch"]),
    )


class Bench:
    """Cues computed once per track; decoding a whole sweep is then nearly free.

    The model forward and the feature load are the expensive parts and do not depend
    on any decoder constant, so hoisting them out is what makes an honest sweep
    affordable rather than something to economise on.
    """

    def __init__(self, ckpt: Path, tag: str, split: str, limit: int) -> None:
        condition = CONDITIONS[canonical(tag)]
        dataset = condition.dataset(ROOT, split, limit)
        validator = CheckpointValidator(dataset, condition, cache=Store(CACHE_DIR))
        encoder = LearnedEncoder(load_cuenet(ckpt))
        self.cues = [
            encoder.encode_features(ex.contour, ex.load_features())
            for ex in validator.examples
        ]
        self.refs = [ex.notes for ex in validator.examples]

    def run(self, **kwargs) -> Scored:
        decoder = GreedyDecoder(**kwargs)
        return score(self.refs, [decoder.decode(c) for c in self.cues])


def show(label: str, s: Scored, base: Scored | None = None) -> None:
    delta = f"{s.strict_f1 - base.strict_f1:+7.4f}" if base else "       "
    print(f"  {label:<22}{s.recall:>8.4f} {s.precision:>10.4f} {s.blind_f1:>9.4f} "
          f"{s.strict_f1:>10.4f} {delta} {s.frag:>7} {s.offpitch:>9}")


def header() -> None:
    print(f"  {'setting':<22}{'recall':>8} {'precision':>10} {'blind F1':>9} "
          f"{'strict F1':>10} {'delta':>7} {'frag':>7} {'offpitch':>9}")


# Each entry: the constant, the values to try, and what it was invented to do. A value
# of None for merge_gap_s means the pass is disabled -- that is already the default.
SWEEPS: list[tuple[str, list, str]] = [
    ("pitch_split_semitones", [0.8, 1.5, 3.0, OFF],
     "place a note where pitch changes -- superseded by the onset head"),
    ("min_note_s", [0.0, 0.01, 0.02, 0.03, 0.05, 0.08],
     "drop tracker blips -- the voicing head should already suppress them"),
    ("min_onset_gap_s", [0.0, 0.01, 0.02, 0.03, 0.05, 0.08],
     "refractory gap so one attack cannot fire twice"),
    ("onset_threshold", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
     "operating point on the onset curve -- a real choice, not a leftover"),
    ("voicing_threshold", [0.3, 0.4, 0.5, 0.6, 0.7],
     "operating point on the voicing curve -- likewise"),
    ("merge_gap_s", [None, 0.01, 0.03, 0.05],
     "rejoin same-pitch fragments -- the over-segmentation lever, already off"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--condition", default="separated")
    p.add_argument("--split", default="validation")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument(
        "--baseline", action="store_true",
        help="sweep against today's shipped defaults instead of pitch-split-off",
    )
    args = p.parse_args()

    bench = Bench(args.checkpoint, args.condition, args.split, args.limit)

    # Everything is swept with the pitch split already removed. Auditing the rest
    # underneath a constant we know is broken would measure its interaction with a
    # bug rather than its own worth.
    fixed = {} if args.baseline else {"pitch_split_semitones": OFF}
    base = bench.run(**fixed)

    print(f"\n{args.checkpoint.stem} on {args.condition} / {args.split} "
          f"({len(bench.refs)} tracks, {sum(len(r) for r in bench.refs)} notes)")
    print(f"reference point: {'shipped defaults' if args.baseline else 'pitch split OFF'}\n")
    header()
    show("baseline", base)

    for name, values, why in SWEEPS:
        if name in fixed:
            continue
        print(f"\n  {name}  -- {why}")
        header()
        for v in values:
            show(f"{name.split('_')[0]}={v}", bench.run(**fixed, **{name: v}), base)


if __name__ == "__main__":
    main()
