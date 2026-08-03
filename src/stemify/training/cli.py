"""Command-line entry point for training.

    uv run stemify-train --root data/raw/slakh2100 \
        --cache-dir data/cache --tracking-uri file:./mlruns \
        --sources clean,separated --pitch

This module is only argument parsing and flag validation; the run itself is
``training.train.train``. Split out for the same reason ``evaluation/cli.py`` is
separate from the evaluation runners: flags change on a different schedule from the
thing they configure, and ``train.py`` was the module that kept attracting whatever
had nowhere else to go.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from stemify.training.conditions import CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.recipes import HEADS, RECIPES
from stemify.training.semicrf.recipe import PITCH_WEIGHT
from stemify.training.train import train


def _parse_pos_weight(spec: str) -> dict[str, float]:
    """``"onset=5,voicing=2"`` -> ``{"onset": 5.0, "voicing": 2.0}``.

    Validated against ``HEADS`` because a typo here fails *silently* in the worst way:
    ``FrameBCELoss`` reads its weights with ``.get(head)``, so an unrecognised key is
    simply never looked up and the run trains unweighted while its logged params claim
    otherwise. That is precisely how ``pos_weight`` came to be declared in
    ``TrainConfig``, plumbed through the loss, and never once set by anything.
    """
    weights: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        head, sep, value = part.partition("=")
        if not sep or head not in HEADS:
            raise ValueError(
                f"--pos-weight expects 'head=value' with head in {HEADS}, got {part!r}"
            )
        weights[head] = float(value)
    return weights


def main() -> None:
    # Configure logging once, here at the entry point -- library modules only ever
    # getLogger, never configure, so importing them stays side-effect free.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/slakh2100")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/onset"))
    ap.add_argument("--sources", default="clean,separated",
                    help="comma-separated train input conditions from "
                         + ", ".join(CONDITIONS) + ". A 'bichannel-' condition feeds "
                         "two channels, the bass first and the mixture it came from "
                         "second, and needs a 2-channel model")
    ap.add_argument("--val-sources",
                    help="comma-separated val conditions, scored+logged separately "
                         "(default: every deployment condition of the model's own "
                         "channel width)")
    ap.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. file:./mlruns")
    ap.add_argument("--experiment", default="onset-detection",
                    help="MLflow experiment name")
    ap.add_argument("--run-name", default=None,
                    help="MLflow run name. Default is derived from the run itself -- "
                         "objective, extra heads, and condition initials -- because a "
                         "fixed default labelled every run of every kind the same.")
    ap.add_argument("--limit", type=int, help="cap train tracks per source")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--batch-log-every", type=int, default=100,
                    help="log a smoothed train loss to MLflow every N steps (0 = off)")
    ap.add_argument("--objective", default="frame", choices=sorted(RECIPES),
                    help="training objective: 'frame' is per-frame BCE over fixed "
                         "windows grouped by the greedy decoder; 'semicrf' trains the "
                         "segmental decoder jointly with the encoder, over ordered "
                         "chunks of a track carrying the alpha window between them")
    ap.add_argument("--init-from", type=Path,
                    help="warm-start the encoder from a checkpoint's state_dict "
                         "(objective parameters are not restored)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--pos-weight", default="",
                    help="per-head BCE positive weight, e.g. 'onset=5'. Comma-separated. "
                         "Onsets are ~7%% of frames after +-1 widening, so >1 trades "
                         "precision for recall -- the candidate-generator setting for a "
                         "two-stage decoder. Unset (the default) is plain unweighted BCE")
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs with no val gain. Worth "
                         "raising for a structured objective, whose val curve is "
                         "noisier than frame BCE's")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader worker subprocesses (0 = main thread)")
    ap.add_argument("--pitch", action="store_true",
                    help="train the learned pitch branch alongside onset/voicing, "
                         "so the encoder produces the contour instead of pyin. Adds a "
                         "per-frame MIDI target to each cached example.")
    ap.add_argument("--pitch-weight", type=float, default=None,
                    help="semicrf + --pitch only: weight on the auxiliary pitch "
                         "cross-entropy in 'NLL + w * pitch_ce'. Defaults to "
                         f"{PITCH_WEIGHT}, which is a starting point rather than a "
                         "tuned value -- the two terms are on unrelated scales.")
    ap.add_argument("--out", type=Path,
                    help="checkpoint path (default <cache-dir>/onset_model.pt)")
    args = ap.parse_args()

    # The recipe is built first because it owns which heads this run trains: with
    # --pitch that is three, and the cached targets and the checkpoint both have to
    # agree with that list.
    #
    # --pos-weight is a frame-strategy setting and is refused elsewhere rather than
    # ignored: the semi-CRF's objective takes no class weights, so while this lived on
    # TrainConfig the flag was accepted and quietly did nothing.
    #
    # --pitch used to be refused here too, because nothing supervised the head's
    # *values*: the span features read how far the distribution moves and how much its
    # frames disagree, neither of which says whether it is right. `SemiCRFRecipe` now
    # declares pitch as an auxiliary head and `SemiCRFLoss` adds
    # `pitch_weight * pitch_ce` over each chunk's own frames, so the head trains on a
    # real signal and the flag means what it does on the frame path.
    strategy_kwargs: dict[str, Any] = {"pitch": args.pitch}
    if args.objective == "frame":
        strategy_kwargs["pos_weight"] = _parse_pos_weight(args.pos_weight)
        if args.pitch_weight is not None:
            raise SystemExit(
                "--pitch-weight applies to --objective semicrf, not 'frame'. The frame "
                "objective sums its head terms unweighted; there is no NLL to balance "
                "the pitch term against."
            )
    else:
        if args.pos_weight:
            raise SystemExit(
                "--pos-weight applies to --objective frame, not "
                f"{args.objective!r}: a structured NLL takes no class weights."
            )
        if args.pitch_weight is not None:
            if not args.pitch:
                raise SystemExit(
                    "--pitch-weight only means something with --pitch: it scales the "
                    "auxiliary pitch term, and without the head there is no such term."
                )
            strategy_kwargs["pitch_weight"] = args.pitch_weight
    recipe = RECIPES[args.objective](args.sources.split(","), **strategy_kwargs)
    config = TrainConfig(
        max_epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, patience=args.patience,
    )
    train(
        args.root, args.cache_dir, config,
        recipe=recipe,
        init_from=args.init_from,
        val_conditions=args.val_sources.split(",") if args.val_sources else None,
        val_limit=args.val_limit, val_every=args.val_every,
        experiment=args.experiment, run_name=args.run_name,
        batch_log_every=args.batch_log_every,
        tracking_uri=args.tracking_uri, limit=args.limit, out=args.out,
    )


if __name__ == "__main__":
    main()
