"""Train the frame heads on Slakh, scored by note-level F1.

    uv run stemify-train --root data/raw/slakh2100 \
        --cache-dir data/cache --tracking-uri file:./mlruns \
        --sources clean,separated --pitch

(``python -m stemify.training.train`` still works; the console script is declared in
``pyproject.toml`` so the entry point is discoverable without reading this file.)

Trains onset and voicing, plus a learned pitch contour with ``--pitch``, which replaces
pyin as the pitch source (see docs/experiments.md for what that is worth).

Domain randomisation: the input is drawn from several *conditions* (``--sources``,
tags from ``training.conditions``) while the target is always the clean-stem MIDI, so
the heads are robust to the condition deployment actually sees. ``clean`` needs no
separation; ``separated`` runs demucs over the train set first, cached.

Validation runs on every deployment condition of the model's own channel width
(``--val-sources``), not just clean stems. Cheap per-frame val loss drives early
stopping inline, while ``AsyncValidationCallback`` scores each checkpoint through the
real ``LearnedEncoder`` + ``HeuristicDecoder`` pair -- one ``OnsetValidator`` per
condition, each caching its contours so a score is a model forward rather than a pyin
re-run. From that one forward it logs both altitudes to MLflow --
``note/<condition>/f1`` (decoder) and ``cues/<condition>/{onset_ap, voicing_auroc}``
(encoder) -- and attaches a per-epoch ``validation/`` JSON artifact with the full
breakdown. The gap between ``note/clean/f1`` and ``note/separated/f1`` is the verdict
on how well the learned cues survive separation.
"""

import argparse
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.contour import Contour
from stemify.decoders.base import Decoder
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.encoders.learned import LearnedEncoder
from stemify.features import FrameFeatures
from stemify.notes import NoteEvent
from stemify.models.cuenet import CueNet
from stemify.training.checkpoint import Checkpoint
from stemify.training.callbacks import (
    AsyncValidationCallback,
    Callback,
    ConsoleCallback,
    MLflowCallback,
)
from stemify.training.conditions import CONDITIONS, canonical
from stemify.training.config import TrainConfig
from stemify.training.recipes import HEADS, RECIPES, Recipe
from stemify.training.semicrf.recipe import PITCH_WEIGHT
from stemify.training.trainer import Trainer
from stemify.training.validation import OnsetValidator

logger = logging.getLogger(__name__)

# score_examples' metric keys -> the short leaf name under note/<condition>/, so
# MLflow groups them: note/clean/f1, note/separated/recall, ... one per metric.
_NOTE_KEY = {
    "note_f1": "f1", "note_precision": "precision",
    "note_recall": "recall", "fragmentation": "fragmentation",
    # The remaining false-positive buckets pass through under their own names
    # (fp, fp_rest, fp_interior_offpitch) -- see ``score_examples``.
}


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


def _epoch_batches(loader: DataLoader[Any]) -> int | str:
    """Batches per epoch, or ``"streaming"`` when that is not knowable in advance.

    The chunked semi-CRF loader wraps an ``IterableDataset``: it walks ``batch_size``
    parallel streams and stops when the first one runs out, so its epoch length is a
    property of how the tracks divided up rather than something ``len`` can answer.
    Logged as a param either way -- it is the number that makes "10 epochs" comparable
    between objectives, and assuming every loader is sized is what crashed a run at
    the first MLflow write while every smoke test (which ran without a tracking URI)
    passed."""
    try:
        return len(loader)
    except TypeError:
        return "streaming"


def build_validators(
    root: str,
    val_limit: int,
    conditions: list[str],
    cache: Cache | None = None,
) -> dict[str, OnsetValidator]:
    """One ``OnsetValidator`` per condition, keyed by tag. A condition names the
    channels the model reads -- exactly what a validator has to reproduce -- so the
    same ``CONDITIONS`` table that defines the *training* conditions defines the
    validation ones.

    Every condition shares the one ``cache``: contours and CQTs are keyed on each
    condition's *resolved* audio path, so they cannot collide and no per-tag
    subdirectory is needed. pyin then runs once per file ever rather than at every
    launch -- what makes a large val set affordable. The demucs condition separates
    its val audio through the (cached) DemucsSeparator at construction.

    The per-track loop is serial, and whether that costs anything **depends on the
    numeric backend**. On macOS/Accelerate pyin's FFTs already spread across cores and
    a process pool only oversubscribes; on Linux/OpenBLAS (scipy's pocketfft is
    single-threaded unless handed ``workers``) it pins one core and a pool measures
    ~4.9x. Serial is the right default because the cache means this is paid once per
    file ever -- but see ``docs/refactor-plan.md`` before concluding a pool is
    pointless on a given machine."""
    return {
        tag: OnsetValidator(
            CONDITIONS[tag].dataset(root, "validation", val_limit),
            CONDITIONS[tag],
            cache=cache,
        )
        for tag in conditions
    }


def _eval_fn(
    validators: dict[str, OnsetValidator],
    report_dir: Path | None = None,
    log_artifact: Callable[[Path, int], None] | None = None,
    build_decoder: Callable[[Mapping[str, Any]], Decoder] | None = None,
    *,
    build_model: Callable[[], CueNet],
) -> Callable[[Mapping[str, Any], int], Mapping[str, float]]:
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights and
    score it through each cached validator (off the training device). Every metric is
    namespaced ``<level>/<condition>/<key>`` so MLflow charts one series per condition
    and the robustness gap between stem and demucs is visible.

    Each checkpoint is scored at both altitudes from the *same* model pass per track:
    note-level (the decoder over the encoder's cues) and cues-level (the encoder's
    onset/voicing head quality, via ``metrics.cues.score_cues`` -- the same core the
    offline ``CuesEvaluation`` uses). If ``report_dir``/``log_artifact`` are given, the
    full breakdown is also written as a per-epoch JSON and attached to the run, so a
    checkpoint's scores survive as a diffable artifact, not only as metric points.

    ``build_decoder(state) -> Decoder`` is the seam the objective varies: the frame
    heads are grouped by the ``HeuristicDecoder``, while a semi-CRF run must be scored
    through the potentials in its own snapshot. Defaulting it to the heuristic keeps
    the frame path unchanged, but hardcoding it would mean a semi-CRF run silently
    reported the *heuristic's* note F1 over its learned cues -- plausible numbers for
    a model nobody trained.

    ``build_model`` is **required and keyword-only**, and is always
    ``recipe.build_model``. It used to default to a module-level ``build_model`` here
    that built a TCN with onset and voicing heads and no pitch branch -- a second,
    weaker implementation of what the recipe already knows. Nothing took the default,
    but anything that did would have validated a model missing the head the run was
    training, and reported its scores as the run's. The recipe is the only thing that
    knows what this run builds, so it is the only thing allowed to build it."""
    if build_decoder is None:
        def build_decoder(_state: Mapping[str, Any]) -> Decoder:
            return HeuristicDecoder()

    def eval_fn(state: Mapping[str, Any], epoch: int) -> Mapping[str, float]:
        model = build_model()  # the recipe's encoder
        # `state` is a Trainer.state_cpu snapshot: the model's weights under "model"
        # and the objective's own parameters under "loss". The decoder is built from
        # the same snapshot, so a structured objective is scored through the
        # potentials that snapshot actually trained -- not a fresh set.
        model.load_state_dict(dict(state["model"]))
        # Wrap the checkpoint once (CPU, off the training device) as the encoder/decoder
        # pair deployment composes into a Segmenter. The decoder scores notes over the
        # encoder's cues; the encoder alone scores its own cues -- one forward, both.
        encoder = LearnedEncoder(model, device="cpu")
        decoder = build_decoder(state)

        def segment(contour: Contour, features: FrameFeatures) -> list[NoteEvent]:
            return decoder.decode(encoder.encode_features(contour, features))

        metrics: dict[str, float] = {}
        report: dict[str, Any] = {"epoch": epoch, "conditions": {}}
        for tag, validator in validators.items():
            note = validator.score(segment)
            for key, value in note.items():
                metrics[f"note/{tag}/{_NOTE_KEY.get(key, key)}"] = value
            # Cues-level (encoder) quality: onset-curve AP and voicing-head AUROC,
            # threshold-free. Empty voicing for an onset-only model.
            cues = {r.name: r.scalars for r in validator.score_cues(encoder.encode_features)}
            for scalars in cues.values():
                for key, value in scalars.items():
                    metrics[f"cues/{tag}/{key}"] = value
            report["conditions"][tag] = {"note": note, "cues": cues}

        if report_dir is not None and log_artifact is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            path = report_dir / f"validation_epoch{epoch:03d}.json"
            path.write_text(json.dumps(report, indent=2))
            log_artifact(path, epoch)
        return metrics

    return eval_fn


def train(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    val_conditions: list[str] | None = None,
    val_limit: int = 40,
    val_every: int = 5,
    tracking_uri: str | None = None,
    experiment: str = "onset-detection",
    run_name: str = "tcn-v2",
    batch_log_every: int = 100,
    limit: int | None = None,
    out: Path | None = None,
    recipe: Recipe | None = None,
    init_from: Path | None = None,
) -> None:
    recipe = recipe or RECIPES["frame"]()
    sources = [c.name for c in recipe.conditions]
    # Validate on every deployment condition (not just clean stems), so the note
    # F1 gap between stem and demucs is visible during training rather than only
    # at final test. Independent of train sources: monitor the full breakdown.
    # The default tracks the model's input width, since a validator has to feed
    # the same channels the model was built for.
    val_conditions = val_conditions or (
        ["bichannel-clean", "bichannel-separated"] if recipe.in_channels == 2
        else ["clean", "mixture", "separated"]
    )
    # Through `canonical` for the same reason `--sources` is: a run given retired tags
    # otherwise resolves its training conditions and then dies looking up the val ones.
    val_conditions = [canonical(t) for t in val_conditions]
    unknown = [t for t in val_conditions if t not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown val condition(s) {unknown}; have {sorted(CONDITIONS)}"
        )
    mismatched = [
        t for t in val_conditions if CONDITIONS[t].in_channels != recipe.in_channels
    ]
    if mismatched:
        raise ValueError(
            f"val conditions {mismatched} do not match the model's "
            f"{recipe.in_channels} input channel(s) (training on {sources}). "
            "Scoring needs the same channel layout the model was trained with."
        )
    cache = Cache(cache_dir)
    out = out or cache_dir / "onset_model.pt"

    # One cache for every condition and both splits. Features are content-addressed on
    # the resolved audio path, so a stem, a mix and a separated mix of the same track
    # land on distinct keys without any tag/split subdirectory to keep them apart --
    # and the val conditions transparently reuse whatever this loop already built.
    # The recipe owns the batch shape: fixed windows for frame BCE, whole tracks for
    # the semi-CRF, whose objective scans a whole segmentation.
    train_loader = recipe.train_loader(root, cache, config, limit)
    val_loader = recipe.val_loader(root, cache, config, val_limit)

    model = recipe.build_model()
    if init_from is not None:
        # Encoder weights only. Every recipe builds the same backbone + heads
        # (SemiCRFRecipe.build_model defers to FrameRecipe's), so a checkpoint from
        # one objective initialises another -- which is how a semi-CRF run gets to
        # start from *good* cues instead of learning the encoder and the potentials
        # simultaneously from scratch. Strict: a shape mismatch should fail loudly,
        # not silently leave half the encoder random.
        warm = Checkpoint.load(init_from)
        model.load_state_dict(warm.model_state)
        if warm.objective_state:
            logger.warning(
                "%s carries objective parameters (%s); --init-from restores the "
                "encoder only, so these are dropped",
                init_from, sorted(warm.objective_state),
            )
        logger.info("initialised encoder from %s", init_from)

    trainer = Trainer(model, recipe.build_loss(), config)

    callbacks: list[Callback] = [ConsoleCallback()]
    mlflow = None
    if tracking_uri:
        # Run names are labels, not keys -- MLflow mints a fresh run_id per launch,
        # so two runs can share a name. Suffix a UTC timestamp so relaunching with
        # the same --run-name yields distinct, self-dating runs ("tcn-v1" stays the
        # groupable prefix) rather than ambiguous duplicates.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        mlflow = MLflowCallback(
            run_name=f"{run_name}-{stamp}", experiment=experiment, tracking_uri=tracking_uri,
            batch_log_every=batch_log_every,
            # Loop-level settings here; strategy knobs come from ``recipe.params``,
            # which is the only thing that knows which of them this run even has.
            params={"encoder": "tcn", "sources": ",".join(sources),
                    "val_conditions": ",".join(val_conditions),
                    "lr": config.learning_rate,
                    "objective": recipe.name,
                    "in_channels": recipe.in_channels,
                    "train_batches": _epoch_batches(train_loader),
                    **recipe.params},
        )
        # Precompute pyin (and, for demucs, the separation) over each val condition
        # once; async scores are then just model forwards. Slowest at construction
        # is demucs -- it separates its val audio before pyin.
        logger.info(
            "caching val contours for %d tracks x conditions %s...",
            val_limit, val_conditions,
        )
        validators = build_validators(root, val_limit, val_conditions, cache=cache)
        # Per-epoch validation reports (all conditions x note+cues) are written here
        # and attached to the run under "validation/", so each scored checkpoint leaves
        # a diffable JSON artifact, not only metric points.
        eval_fn = _eval_fn(
            validators,
            report_dir=cache_dir / "val_reports",
            log_artifact=lambda p, _e: mlflow.log_artifact(p, artifact_path="validation"),
            # Without these a semi-CRF run would be scored as a freshly-initialised
            # HeuristicDecoder over its learned cues: plausible numbers, wrong model.
            build_decoder=recipe.decoder_from,
            build_model=recipe.build_model,
        )
        callbacks += [
            mlflow,
            AsyncValidationCallback(eval_fn, mlflow.sink, every=val_every),
        ]

    trainer.fit(train_loader, val_loader, callbacks=callbacks)

    # fit() restores the best-val checkpoint into trainer.model; persist it so the
    # trained heads survive the process (a LearnedEncoder loads them) rather than
    # living only as MLflow metrics. Save the state_dict plus the metadata needed to
    # rebuild the architecture before loading weights.
    # The recipe goes in rather than the facts derived from it (heads, in_channels,
    # the decoder to score through): recording the recipe means a reader rebuilds them
    # the same way training did, instead of re-deriving them and getting it subtly
    # wrong. The objective's own parameters travel too -- for a semi-CRF they *are*
    # the decoder, and a file without them restores half a system while looking whole.
    Checkpoint(
        recipe=recipe,
        model_state=trainer.model.state_dict(),
        objective_state=trainer.loss.state_dict(),
    ).save(out)
    logger.info("saved onset model -> %s", out)
    if mlflow is not None:
        mlflow.log_artifact(out)


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
    ap.add_argument("--run-name", default="tcn-v2", help="MLflow run name")
    ap.add_argument("--limit", type=int, help="cap train tracks per source")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--batch-log-every", type=int, default=100,
                    help="log a smoothed train loss to MLflow every N steps (0 = off)")
    ap.add_argument("--objective", default="frame", choices=sorted(RECIPES),
                    help="training objective: 'frame' is per-frame BCE over fixed "
                         "windows grouped by the heuristic; 'semicrf' trains the "
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