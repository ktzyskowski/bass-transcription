"""Is the semi-CRF's recall collapse a decode bias, or a worse model?

A run whose val NLL, onset AP and voicing AUROC all improved every epoch emitted 13,516
notes at epoch 0, 9,310 at epoch 1 and 8,783 at epoch 2, against 14,238 in the
references. Precision barely moved and fragmentation fell with it, so the decoder was
not trading risky notes away -- it was dropping true positives faster than false ones,
which is what a granularity knob looks like, not a judgement.

The candidate explanation is the gap between the objective and the readout. Training
minimises ``log Z - score(gold)``, and ``log Z`` sums over every segmentation, so it
accounts for the combinatorially many ways to place ``k`` boundaries. Viterbi maximises
over one segmentation at a time: it pays the per-span cost that multiplicity justified
and never sees the multiplicity. The best single path is therefore coarser than the
distribution's own expected note count, and gets coarser as the model sharpens.

If that is what happened, one scalar reproduces it. Sweeping ``note_bias`` **negative**
on the epoch-0 checkpoint should trace out the epoch-1 and epoch-2 points -- same note
count, precision holding, fragmentation collapsing -- because they lie on the same
one-dimensional family. Sweeping it positive should recover count, and F1 should turn
over once the bias starts inventing boundaries where the evidence never was ambiguous.

What the shape means, and why the answer is not just the argmax:

* **F1 rises then falls, count crossing the reference near the peak** -- confirmed. The
  correction is real and a flat bias is a serviceable form of it.
* **F1 turns over early, well short of the reference count** -- the correction is real
  but wrongly *shaped*. Theory says the per-segment bonus should be about
  ``log(mean segment length)``, i.e. shrinking as segmentation gets finer, where a flat
  bias does not. The next step is then the duration buckets, not a bigger constant.
* **F1 flat or falling from zero** -- the diagnosis is wrong, and neither this nor
  marginal decoding is worth building.

Selection on validation only. A bias chosen here is fitted to these tracks; quoting it
means re-running the chosen value on held-out test, exactly as ``decoder_audit`` does.
"""

import argparse
import time
from pathlib import Path

import torch

from stemify.cache import Cache
from stemify.encoders.learned import LearnedEncoder
from stemify.metrics import score_notes
from stemify.training.checkpoint import Checkpoint
from stemify.training.validation import build_validators

ROOT = "data/raw/slakh2100"
CACHE_DIR = "data/cache"
# Encode on the accelerator, decode on the host. The segmental DP is launch-bound --
# its cost is kernel count, not arithmetic -- so on this box it measures ~4x faster on
# CPU (1.4 s/track) than on the 5090 (5.8 s/track), while the encoder is ~7x faster on
# the GPU. Splitting them is what makes a nine-point sweep minutes rather than an hour.
ENCODE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DECODE_DEVICE = "cpu"


def sweep(
    checkpoint: Path, root: str, cache_dir: str, conditions: list[str],
    limit: int, biases: list[float], marginal: str = "",
    tau: float = 0.5, split: str = "validation", tol_frames: int = 0,
) -> None:
    ck = Checkpoint.load(checkpoint)
    model = ck.recipe.build_model()
    model.load_state_dict(ck.model_state)
    decoder = ck.recipe.decoder_from({"loss": ck.objective_state})
    decoder.to(torch.device(DECODE_DEVICE))
    encoder = LearnedEncoder(model, device=ENCODE_DEVICE)
    print(f"{checkpoint}  pitch_source={encoder.pitch_source}  "
          f"encode={ENCODE_DEVICE} decode={DECODE_DEVICE}")

    validators = build_validators(root, limit, conditions, cache=Cache(cache_dir), split=split)
    for tag, validator in validators.items():
        examples = validator.examples
        refs = [ex.notes for ex in examples]
        gold = sum(len(r) for r in refs)

        # Encoded once and held: the sweep varies only the readout, so re-encoding per
        # bias would spend all its time recomputing the one thing that cannot change.
        t0 = time.perf_counter()
        cues = [encoder.encode_features(ex.contour, ex.load_features()) for ex in examples]
        print(f"\n=== {tag}: {len(examples)} tracks, {gold} reference notes "
              f"(encoded in {time.perf_counter() - t0:.0f}s) ===")
        print(f"{'bias':>7} {'emitted':>8} {'/gold':>6} {'f1':>7} {'prec':>7} "
              f"{'recall':>7} {'frag':>6} {'rest':>6} {'offp':>6} {'s':>5}")

        # `None` marks the marginal decode: no bias at all, because it corrects
        # per span rather than per track. Scored through the identical path so the
        # comparison is the decision rule and nothing else.
        for bias in [*biases, *([None] if marginal else [])]:
            t0 = time.perf_counter()
            ests = (
                [decoder.decode_marginal(c, marginal, tau, tol_frames) for c in cues]
                if bias is None
                else [decoder.decode(c, note_bias=bias) for c in cues]
            )
            scalars = {r.name: r.scalars for r in score_notes(refs, ests)}
            prf, frag = scalars["note-prf"], scalars["fragmentation"]
            emitted = sum(len(e) for e in ests)
            label = f"MEA:{marginal[:4]}" if bias is None else f"{bias:>7.2f}"
            print(f"{label:>7} {emitted:>8d} {emitted / gold:>6.2f} "
                  f"{prf['f1']:>7.4f} {prf['precision']:>7.4f} {prf['recall']:>7.4f} "
                  f"{frag['fragmentation']:>6.0f} {frag['rest']:>6.0f} "
                  f"{frag['interior_offpitch']:>6.0f} "
                  f"{time.perf_counter() - t0:>5.0f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("data/cache/semicrf_pitch_aux.pt"))
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--conditions", default="clean,separated")
    p.add_argument("--limit", type=int, default=40,
                   help="val tracks per condition; 40 matches the training run's set")
    p.add_argument("--tol-frames", type=int, default=0,
                   help="sum boundary posteriors over +-N frames -- the note "
                        "matcher tolerates 50ms, ~1.4 DP frames, so scoring an "
                        "exact position asks for certainty nothing measures")
    p.add_argument("--split", default="validation", choices=["validation", "test"],
                   help="test is for re-running an ALREADY CHOSEN setting; "
                        "sweeping on it is how a held-out set stops being one")
    p.add_argument("--tau", type=float, default=0.5,
                   help="posterior a span/boundary must beat to be worth "
                        "including; 0.5 is what the symmetric-difference "
                        "loss implies")
    p.add_argument("--marginal", default="", choices=["", "span", "boundary"],
                   help="also decode from span posteriors (MEA) and report it "
                        "as a row labelled MEA -- no bias, since it corrects per "
                        "span rather than per track")
    p.add_argument("--biases", default="-1.5,-1.0,-0.5,-0.25,0,0.25,0.5,1.0,1.5",
                   help="comma-separated; 0 must be present as the baseline that "
                        "reproduces the run's own reported numbers")
    args = p.parse_args()
    sweep(
        args.checkpoint, args.root, args.cache_dir,
        args.conditions.split(","), args.limit,
        [float(b) for b in args.biases.split(",")], args.marginal, args.tau, args.split, args.tol_frames,
    )


if __name__ == "__main__":
    main()
