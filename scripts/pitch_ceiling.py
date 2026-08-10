"""What would a perfect pitch head buy? Score the same notes with pitch ignored.

The measurement that decides between the two stories I have told about the demucs
gap:

  - "pitch is the lever"   -- predicts F1 jumps a lot when pitch stops being matched
  - "the decoder is the lever" -- predicts it barely moves, because the notes are in
    the wrong PLACE, not merely at the wrong pitch

mir_eval matches a note on onset AND pitch, so relaxing pitch_tolerance to something
enormous turns it into onset-only matching. The gap between the two F1s is exactly the
share of errors attributable to pitch, holding the segmentation fixed.
"""
import sys
from pathlib import Path

from stemify.persistence.store import Store
from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.training.checkpoint import load_framenet
from stemify.metrics.note.fragmentation import Fragmentation
from stemify.metrics.note.prf import PRF
from stemify.metrics.note.scoring import score_notes
from stemify.conditions import CONDITIONS
from stemify.training.validation import CheckpointValidator

ROOT = "/home/ktzys/dev/tabgen/stemify-ml/data/raw/slakh2100"
CACHE = Store("/home/ktzys/dev/tabgen/stemify-ml/data/cache")
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only

print(f"{'condition':12} {'F1':>8} {'F1 no-pitch':>12} {'gain':>8} {'pitch share':>12}")
for tag, ckpt in [a.split("=") for a in sys.argv[1:]]:
    condition = CONDITIONS[tag]
    model = load_framenet(Path(ckpt))
    validator = CheckpointValidator(
        condition.dataset(ROOT, "validation", 20), condition, cache=CACHE
    )
    encoder, decoder = LearnedEncoder(model, device="cpu"), GreedyDecoder()

    refs, ests = [], []
    for ex in validator.examples:
        ests.append(decoder.decode(encoder.encode_features(ex.contour, ex.load_features())))
        refs.append(ex.notes)

    strict = {r.name: r.scalars for r in score_notes(refs, ests)}["note-prf"]
    loose = {
        r.name: r.scalars
        for r in score_notes(
            refs, ests, metrics=(PRF(pitch_tolerance=IGNORE_PITCH),
                                 Fragmentation(pitch_tolerance=IGNORE_PITCH))
        )
    }["note-prf"]
    gain = loose["f1"] - strict["f1"]
    # Of everything still missing at F1=1.0, how much is pitch alone?
    share = gain / (1.0 - strict["f1"]) if strict["f1"] < 1.0 else 0.0
    print(f"{tag:12} {strict['f1']:>8.4f} {loose['f1']:>12.4f} {gain:>+8.4f} {share:>11.1%}")
