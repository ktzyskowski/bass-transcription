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
from stemify.training.semicrf.recipe import (
    ONSET_WEIGHT,
    PITCH_WEIGHT,
    VOICING_WEIGHT,
)
from stemify.training.semicrf.terms import AUX_TERMS
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


def _parse_harmonics(spec: str) -> list[int]:
    """``"1,2,3,4"`` -> ``[1, 2, 3, 4]``, the harmonics each CQT channel is stacked at.

    Refuses anything but positive integers *including 1*. Dropping the fundamental is
    representable and almost certainly a mistake -- the pitch head reads this axis, and
    a stack starting at ``h=2`` shifts every bin it decodes by an octave without
    changing a single shape, so nothing downstream would raise.
    """
    values = [int(p) for p in filter(None, (p.strip() for p in spec.split(",")))]
    if not values:
        return [1]
    if any(v < 1 for v in values) or values[0] != 1:
        raise ValueError(
            f"--harmonics must be positive and start at 1 (the fundamental), got {spec!r}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"--harmonics has repeats: {spec!r}")
    return values


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
    ap.add_argument("--harmonics", default="1",
                    help="objective frame only: stack each CQT channel against itself "
                         "at these harmonics, e.g. '1,2,3,4'. A note at bin b also "
                         "sounds at b+36, b+57, b+72, which a 3x3 front-end can never "
                         "reach; stacking puts them at one position. '1' (the default) "
                         "is the plain CQT and changes nothing")
    ap.add_argument("--monitor", default=TrainConfig.monitor,
                    help="which metric picks the best checkpoint and drives early "
                         "stopping. 'loss/<head>/val' is an inline validation loss; "
                         "anything else (note/separated/f1, cues/clean/onset_ap, ...) "
                         "is scored by the harness, which then runs *synchronously* "
                         "because a checkpoint cannot be chosen on a number that does "
                         "not exist yet. The default is the cheap proxy and has been "
                         "caught disagreeing with note F1 by half its value")
    ap.add_argument("--monitor-mode", default=None, choices=["min", "max"],
                    help="whether lower or higher is better. Inferred from --monitor "
                         "when unset: losses minimise, f1/ap/auroc maximise")
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs with no val gain. Worth "
                         "raising for a structured objective, whose val curve is "
                         "noisier than frame BCE's")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accumulate-steps", type=int,
                    default=TrainConfig.accumulate_steps,
                    help="micro-batches per optimiser step. For --objective semicrf "
                         "this is a correctness dial, not a memory one: the alpha "
                         "carry is computed under the weights of the step that made "
                         "it, so moving the weights every step leaves the state a "
                         "chunk consumes ~53 updates stale by a track's end. Holding "
                         "them still for N batches costs nothing and cuts that N-fold, "
                         "at N-fold fewer updates per epoch")
    ap.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate,
                    help="Adam learning rate, over the model and the objective's own "
                         "parameters alike. The default came from the frame path; a "
                         "structured NLL trains the encoder and the potentials jointly "
                         "and is not obliged to want the same one")
    ap.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm,
                    help="clip the total gradient norm over the model and the "
                         "objective's parameters. A safety net against the single bad "
                         "step that sends the weights to inf -- not a tuned "
                         "regulariser, so it should sit above the run's ordinary norm "
                         "(logged per epoch as 'grad_norm'). 0 disables it")
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
    ap.add_argument("--voicing-weight", type=float, default=None,
                    help="semicrf only: weight on the auxiliary voicing BCE. The "
                         "structured NLL supervises voicing only competitively -- it is "
                         "redundant given onset evidence and the duration prior, so the "
                         "scorer can shrink its weight to zero and orphan the head "
                         f"(measured AUROC 0.516 vs frame BCE's 0.984). Default "
                         f"{VOICING_WEIGHT}; 0 restores the structured-only objective.")
    ap.add_argument("--onset-weight", type=float, default=None,
                    help="semicrf only: weight on the auxiliary onset BCE. Measured, "
                         "the structured NLL leaves onset AP at 0.20-0.53 where frame "
                         "BCE reaches 0.994 for the same head -- the larger of the two "
                         f"binary gaps. Default {ONSET_WEIGHT}; 0 restores the "
                         "structured-only objective for this head.")
    ap.add_argument("--out", type=Path,
                    help="where to write this run's checkpoint. No default and no "
                         "fallback: without it the run trains, logs and scores but "
                         "persists nothing, which is what a smoke test wants. There "
                         "used to be a shared default and it overwrote two "
                         "still-cited checkpoints")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists. Checked before "
                         "training starts, not at save time")
    ap.add_argument("--save-epochs", action="store_true",
                    help="also write <out>.epoch<N>.pt after every epoch. Selection "
                         "happens on a decode, so an epoch can be discarded for how it "
                         "was read rather than what it learned -- keep them all and "
                         "choose offline, where the readout can change without "
                         "re-training.")
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
        strategy_kwargs["harmonics"] = _parse_harmonics(args.harmonics)
        for flag, value in (("--pitch-weight", args.pitch_weight),
                            ("--voicing-weight", args.voicing_weight),
                            ("--onset-weight", args.onset_weight)):
            if value is not None:
                raise SystemExit(
                    f"{flag} applies to --objective semicrf, not 'frame'. The frame "
                    "objective sums its head terms unweighted; there is no NLL to "
                    "balance an auxiliary term against."
                )
    else:
        # This used to refuse --pos-weight outright, on the then-true grounds that a
        # structured NLL takes no class weights. It does now: the auxiliary onset and
        # voicing terms are the *same* BCE the frame path runs, so the same imbalance
        # applies and the same knob corrects it.
        if args.pos_weight:
            # Asked of the term rather than compared against the string "pitch". That
            # comparison was a second copy of a fact `AUX_TERMS` already holds, and it
            # would keep passing for a *second* categorical head while doing nothing.
            # `SemiCRFLoss` repeats the check -- a CLI is not the only caller -- but
            # raising here is the difference between a usage error and a traceback.
            weights = _parse_pos_weight(args.pos_weight)
            unweightable = sorted(
                h for h in weights
                if h in AUX_TERMS and not AUX_TERMS[h].takes_pos_weight
            )
            if unweightable:
                raise SystemExit(
                    f"--pos-weight is a *binary* class weight, and {unweightable} "
                    "is scored by a categorical cross-entropy over CQT bins, which "
                    "has no positive class. Use --pitch-weight to scale that term."
                )
            strategy_kwargs["pos_weight"] = weights
        if args.pitch_weight is not None:
            if not args.pitch:
                raise SystemExit(
                    "--pitch-weight only means something with --pitch: it scales the "
                    "auxiliary pitch term, and without the head there is no such term."
                )
            strategy_kwargs["pitch_weight"] = args.pitch_weight
        if args.voicing_weight is not None:
            strategy_kwargs["voicing_weight"] = args.voicing_weight
        if args.onset_weight is not None:
            strategy_kwargs["onset_weight"] = args.onset_weight
    recipe = RECIPES[args.objective](args.sources.split(","), **strategy_kwargs)
    config = TrainConfig(
        max_epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, patience=args.patience,
        learning_rate=args.learning_rate,
        monitor=args.monitor,
        # Inferred rather than required: every metric we log is either a loss (lower
        # is better) or a score in [0,1] (higher is). Making it explicit-or-nothing
        # would mean a run that silently early-stops backwards on a typo'd mode.
        monitor_mode=args.monitor_mode or (
            "min" if args.monitor.startswith("loss/") else "max"
        ),
        accumulate_steps=args.accumulate_steps,
        # 0 rather than a bare --no-clip flag: the value and its off switch stay one
        # thing, so the run's params record "clipping was off" as a value rather than
        # as an absent flag nobody logged.
        max_grad_norm=args.max_grad_norm or None,
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
        save_epochs=args.save_epochs, force=args.force,
    )


if __name__ == "__main__":
    main()
