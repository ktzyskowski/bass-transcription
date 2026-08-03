"""Does bichannel's cue advantage convert once pitch is no longer the bottleneck?

Bichannel has been a null twice, but never on its merits. Under pyin it bought +3.3
recall and +1.5 pitch-blind F1 at equal precision -- a real gain in *placement* -- and
gave it all back through a pitch gap ~1.8 points wider than mono's, because the extra
notes it recovers are the quiet ones pyin handles worst. Now that the learned head has
cut the pitch gap, that trade may not hold.

Answering it properly needs a 2-channel model with a pitch head, which needs ~75 min of
training. This screens it first, for ~10 min and no training, by splitting the two jobs
across the checkpoints we already have:

    onset + voicing   <- a 2-channel model (reads bass AND mix)
    pitch contour     <- rung4-pitch, 1-channel (reads channel 0 only)

Legitimate because channel 0 *is* the bass both models transcribe: the pitch source
sees exactly the audio it would see in a joint model. Holding it fixed across arms
makes the mix channel the only thing that varies, which is the mechanism question.

**What this cannot answer.** A jointly trained 2ch+pitch model may beat the hybrid (a
shared trunk could learn pitch-relevant structure from the mix channel) or trail it.
So this is a decision procedure for whether to spend the GPU time, not the final
number. A clear null here means bichannel is done; a clear gain justifies the arm.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from stemify.cache import Cache
from stemify.cues import Cues
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.encoders.learned import LearnedEncoder
from stemify.models.heads import CONTOUR_VOICING_THRESHOLD
from stemify.training.checkpoint import load_cuenet
from stemify.metrics.note.base import matched_pairs
from stemify.metrics.note.prf import prf_rates
from stemify.models.heads import decode_contour
from stemify.training.conditions import CONDITIONS, canonical
from stemify.training.validation import OnsetValidator

ROOT = "data/raw/slakh2100"
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only

# (arm, checkpoint, the condition whose channel layout it was trained for)
ARMS = [
    ("mono", "rung0-mono", "separated"),
    ("bichannel", "balanced", "bichannel-separated"),
    ("bichannel-w5", "recall-w5", "bichannel-separated"),
]


@torch.no_grad()
def pitch_contour(pitch_model, features, frame_times):
    """A contour from the pitch model, over channel 0 of whatever it is handed.

    Channel 0 is load-bearing across the whole codebase: it is the audio being
    transcribed, and the one every target describes. So slicing it out of a 2-channel
    stack hands the 1-channel pitch model exactly the rendering it was trained on --
    and exactly what a joint model's pitch head would have read.
    """
    values = features.values
    channel0 = values[0] if values.ndim == 3 else values
    x = torch.from_numpy(np.ascontiguousarray(channel0)).float()
    out = pitch_model(x[None, None])                    # (1, 1, F, T)
    logits = out["pitch"].squeeze(0).numpy()
    voiced = None
    if "voicing" in out:
        v = torch.sigmoid(out["voicing"]).squeeze(-1).squeeze(0).numpy()
        voiced = v > CONTOUR_VOICING_THRESHOLD
    return decode_contour(logits, frame_times, voiced)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pitch-checkpoint", type=Path,
                   default=Path("data/checkpoints/rung4-pitch.pt"))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    pitch_model = load_cuenet(args.pitch_checkpoint).eval()
    if "pitch" not in getattr(pitch_model, "frequency_heads", {}):
        raise SystemExit(f"{args.pitch_checkpoint} has no pitch head")
    decoder, cache = HeuristicDecoder(), Cache("data/cache")

    print(f"\npitch source: {args.pitch_checkpoint.stem} (held constant across arms)\n")
    print(f"{'arm':<14} {'split':<11} {'pitch':<9} {'recall':>8} {'prec':>7} "
          f"{'blind F1':>9} {'strict F1':>10}")
    for arm, name, tag in ARMS:
        condition = CONDITIONS[canonical(tag)]
        cue_model = load_cuenet(Path(f"data/checkpoints/{name}.pt"))
        for split in args.splits.split(","):
            v = OnsetValidator(
                condition.dataset(ROOT, split, args.limit), condition, cache=cache
            )
            # "pyin" reproduces the already-recorded baseline on the same track set;
            # "learned" is the same cues with the pitch source swapped underneath.
            for source in ("pyin", "learned"):
                enc = LearnedEncoder(cue_model, device="cpu")
                tp = fp = fn = stp = sfp = sfn = 0
                for ex in v.examples:
                    feats = ex.load_features()
                    cues = enc.encode_features(ex.contour, feats)
                    if source == "learned":
                        cues = Cues(
                            contour=pitch_contour(pitch_model, feats, cues.frame_times),
                            onset=cues.onset,
                            frame_times=cues.frame_times,
                            voicing=cues.voicing,
                        )
                    est = decoder.decode(cues)
                    m = len(matched_pairs(ex.notes, est, pitch_tolerance=IGNORE_PITCH))
                    tp, fp, fn = tp + m, fp + len(est) - m, fn + len(ex.notes) - m
                    s = len(matched_pairs(ex.notes, est))
                    stp, sfp, sfn = stp + s, sfp + len(est) - s, sfn + len(ex.notes) - s
                b, st = prf_rates(tp, fp, fn), prf_rates(stp, sfp, sfn)
                print(f"{arm:<14} {split:<11} {source:<9} {b['recall']:>8.4f} "
                      f"{b['precision']:>7.4f} {b['f1']:>9.4f} {st['f1']:>10.4f}")


if __name__ == "__main__":
    main()
