"""Train the frame heads on Slakh, scored by note-level F1.

    uv run stemify-train --root data/raw/slakh2100 \
        --store-dir data/store --tracking-uri file:./mlruns \
        --sources clean,separated --pitch

(``python -m stemify.training.train`` still works; the console script is declared in
``pyproject.toml`` so the entry point is discoverable without reading this file.)

Trains onset and voicing, plus a learned pitch contour with ``--pitch``, which replaces
pyin as the pitch source (see docs/experiments.md for what that is worth).

Domain randomisation: the input is drawn from several *conditions* (``--sources``,
tags from ``core.conditions``) while the target is always the clean-stem MIDI, so
the heads are robust to the condition deployment actually sees. ``clean`` needs no
separation; ``separated`` runs demucs over the train set first, cached.

Validation runs on every deployment condition of the model's own channel width
(``--val-sources``), not just clean stems. Cheap per-frame val loss drives early
stopping inline, while ``AsyncValidationCallback`` scores each checkpoint through the
real ``LearnedEncoder`` + ``GreedyDecoder`` pair -- one ``CheckpointValidator`` per
condition, each caching its contours so a score is a model forward rather than a pyin
re-run. From that one forward it logs both altitudes to MLflow --
``note/<condition>/f1`` (decoder) and ``cues/<condition>/{onset_ap, voicing_auroc}``
(encoder) -- and attaches a per-epoch ``validation/`` JSON artifact with the full
breakdown. The gap between ``note/clean/f1`` and ``note/separated/f1`` is the verdict
on how well the learned cues survive separation.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from stemify.persistence.store import Store
from stemify.training.checkpoint import Checkpoint
from stemify.training.callbacks import (
    AsyncValidationCallback,
    BestCheckpointCallback,
    EveryEpochCheckpointCallback,
    Callback,
    ConsoleCallback,
    MLflowCallback,
)
from stemify.conditions import CONDITIONS, canonical
from stemify.fitting.config import TrainConfig
from stemify.fitting.base import epoch_batches
from stemify.training.recipes import HEADS, RECIPES, Recipe
from stemify.training.trainer import Trainer
from stemify.training.validation import build_eval_fn, build_validators

logger = logging.getLogger(__name__)




def train(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    val_conditions: list[str] | None = None,
    val_limit: int = 40,
    val_every: int = 5,
    tracking_uri: str | None = None,
    experiment: str = "onset-detection",
    run_name: str | None = None,
    batch_log_every: int = 100,
    limit: int | None = None,
    out: Path | None = None,
    recipe: Recipe | None = None,
    init_from: Path | None = None,
    save_epochs: bool = False,
    force: bool = False,
) -> None:
    recipe = recipe or RECIPES["frame"]()
    sources = [c.name for c in recipe.conditions]
    # Validate on every deployment condition (not just clean stems), so the note
    # F1 gap between stem and demucs is visible during training rather than only
    # at final test. Independent of train sources: monitor the full breakdown.
    # The default tracks the model's input width, since a validator has to feed
    # the same channels the model was built for.
    #
    # `mixture` is deliberately *not* in the default. It is the un-separated negative
    # control, and it has been settled since July -- bass buried in a full mix scores
    # 0.18-0.27 whatever the model, which is the finding that made separation
    # non-negotiable. Scoring it every epoch costs a third of validation to re-derive a
    # conclusion no run has moved. It is still one flag away (`--val-sources
    # clean,mixture,separated`) for when the question is reopened. The bichannel
    # default already omitted its equivalent, so this also makes the two symmetric.
    val_conditions = val_conditions or (
        ["bichannel-clean", "bichannel-separated"] if recipe.in_channels == 2
        else ["clean", "separated"]
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
    store = Store(cache_dir)
    # **No default path, and the check runs before any work starts.**
    #
    # There was a default (`<store-dir>/onset_model.pt`), and it was wrong twice over.
    # Wrong as a *name*, because it stopped describing the model when a voicing head
    # arrived and stopped describing it three ways over once pitch did. Wrong as a
    # *default*, because every run ever launched without `--out` wrote the same file:
    # tcn-v1 was overwritten by tcn-v2, a warning went into docs/experiments.md, and
    # tcn-v2 was then overwritten by a two-track smoke run anyway. A shared default for
    # a per-run artefact is a collision waiting on a careless launch.
    #
    # So `out=None` now means *do not save*, which is what a smoke test actually wants,
    # and naming a run's checkpoint is a deliberate act. The existence check is here
    # rather than at save time because a run that discovers its path is occupied after
    # hours of training can only clobber or be thrown away.
    if out is not None and out.exists() and not force:
        raise FileExistsError(
            f"{out} already exists. Name this run's own checkpoint, or pass --force to "
            "overwrite. A checkpoint is the only artefact a result cannot be reproduced "
            "without, so overwriting one is never implicit."
        )
    if save_epochs and out is None:
        raise ValueError(
            "--save-epochs needs --out: it writes '<stem>.epoch<N><suffix>' beside the "
            "run's checkpoint, and there is nowhere to put those without one."
        )

    # One store for every condition and both splits. Features are content-addressed on
    # the resolved audio path, so a stem, a mix and a separated mix of the same track
    # land on distinct keys without any tag/split subdirectory to keep them apart --
    # and the val conditions transparently reuse whatever this loop already built.
    # The recipe owns the batch shape: fixed windows for frame BCE, whole tracks for
    # the semi-CRF, whose objective scans a whole segmentation.
    train_loader = recipe.train_loader(root, store, config, limit)
    val_loader = recipe.val_loader(root, store, config, val_limit)

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

    # A harness metric (note/*, cues/*) can only drive selection if it is scored on the
    # critical path: the async validator delivers epoch N's score partway through epoch
    # N+1, by which time N's checkpoint decision is made. So the monitor key decides
    # which way the harness runs, rather than being a second flag that can disagree
    # with it.
    sync_scoring = not config.monitor.startswith("loss/")
    if sync_scoring and val_every != 1:
        raise ValueError(
            f"--monitor {config.monitor!r} is scored by the harness, so it must run "
            f"every epoch, but --val-every is {val_every}. On a skipped epoch the "
            "monitored metric would not exist and the run would die mid-training."
        )

    # Written on every improvement, not once at the end: `fit` restores the best
    # weights when it *returns*, so a Ctrl-C, a SIGKILL or an OOM loses the whole run.
    callbacks: list[Callback] = [ConsoleCallback()]
    if out is not None:
        callbacks.append(BestCheckpointCallback(out, recipe))
        if save_epochs:
            callbacks.append(EveryEpochCheckpointCallback(out, recipe))
    mlflow = None
    eval_fn = None
    if tracking_uri:
        # Run names are labels, not keys -- MLflow mints a fresh run_id per launch,
        # so two runs can share a name. Suffix a UTC timestamp so relaunching with
        # the same --run-name yields distinct, self-dating runs ("tcn-v1" stays the
        # groupable prefix) rather than ambiguous duplicates.
        # Derived from what the run *is* when the caller did not name it. The old
        # default was the literal string "tcn-v2" -- a specific frame checkpoint in
        # docs/experiments.md -- so a semi-CRF run with a pitch head logged itself
        # under the name of an unrelated model, and every run of every kind collided
        # in the same label. A run's identity is its objective, its heads and its
        # conditions; those are exactly what makes two runs comparable or not.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        name = run_name or "-".join(
            [recipe.name, *sorted(set(recipe.heads) - set(HEADS)),
             "".join(sorted(c.name[0] for c in recipe.conditions))]
        )
        mlflow = MLflowCallback(
            run_name=f"{name}-{stamp}", experiment=experiment, tracking_uri=tracking_uri,
            batch_log_every=batch_log_every,
            # Loop-level settings here; strategy knobs come from ``recipe.params``,
            # which is the only thing that knows which of them this run even has.
            # ``pitch_source`` mirrors what the validator records per epoch, hoisted to
            # a param so the UI answers "was this scored on the learned head or pyin?"
            # without opening an artifact. Derived from the heads this run trains,
            # which is what decides whether the encoder has a branch to read.
            params={"encoder": "tcn", "sources": ",".join(sources),
                    "pitch_source": "learned" if "pitch" in recipe.heads else "pyin",
                    "val_conditions": ",".join(val_conditions),
                    "lr": config.learning_rate,
                    "batch_size": config.batch_size,
                    "accumulate_steps": config.accumulate_steps,
                    "max_grad_norm": config.max_grad_norm,
                    "objective": recipe.name,
                    "in_channels": recipe.in_channels,
                    "train_batches": epoch_batches(train_loader),
                    **recipe.params},
        )
        # Precompute pyin (and, for demucs, the separation) over each val condition
        # once; async scores are then just model forwards. Slowest at construction
        # is demucs -- it separates its val audio before pyin.
        logger.info(
            "caching val contours for %d tracks x conditions %s...",
            val_limit, val_conditions,
        )
        validators = build_validators(root, val_limit, val_conditions, store=store)
        # Per-epoch validation reports (all conditions x note+cues) are written here
        # and attached to the run under "validation/", so each scored checkpoint leaves
        # a diffable JSON artifact, not only metric points.
        eval_fn = build_eval_fn(
            validators,
            report_dir=cache_dir / "val_reports",
            log_artifact=lambda p, _e: mlflow.log_artifact(p, artifact_path="validation"),
            # Without these a semi-CRF run would be scored as a freshly-initialised
            # GreedyDecoder over its learned cues: plausible numbers, wrong model.
            build_decoder=recipe.decoder_from,
            build_model=recipe.build_model,
        )
        callbacks.append(mlflow)
        if not sync_scoring:
            # Off the critical path, delivering to its own sink. Only sound when the
            # score is not what picks the checkpoint.
            callbacks.append(
                AsyncValidationCallback(eval_fn, mlflow.sink, every=val_every)
            )

    if sync_scoring and eval_fn is None:
        # Reachable: --monitor names a harness metric on a run with no tracking URI, so
        # nothing built the validators. Failing here beats failing at the first epoch
        # boundary an hour in.
        logger.info(
            "caching val contours for %d tracks x conditions %s (for --monitor %s)...",
            val_limit, val_conditions, config.monitor,
        )
        eval_fn = build_eval_fn(
            build_validators(root, val_limit, val_conditions, store=store),
            build_decoder=recipe.decoder_from,
            build_model=recipe.build_model,
        )

    trainer.fit(
        train_loader, val_loader, callbacks=callbacks,
        score_fn=eval_fn if sync_scoring else None,
    )

    # fit() restores the best-val checkpoint into trainer.model; persist it so the
    # trained heads survive the process (a LearnedEncoder loads them) rather than
    # living only as MLflow metrics. Save the state_dict plus the metadata needed to
    # rebuild the architecture before loading weights.
    # The recipe goes in rather than the facts derived from it (heads, in_channels,
    # the decoder to score through): recording the recipe means a reader rebuilds them
    # the same way training did, instead of re-deriving them and getting it subtly
    # wrong. The objective's own parameters travel too -- for a semi-CRF they *are*
    # the decoder, and a file without them restores half a system while looking whole.
    if out is None:
        logger.info("no --out given: trained weights are not persisted")
        return
    Checkpoint(
        recipe=recipe,
        model_state=trainer.model.state_dict(),
        objective_state=trainer.loss.state_dict(),
    ).save(out)
    logger.info("saved checkpoint -> %s", out)
    if mlflow is not None:
        mlflow.log_artifact(out)


if __name__ == "__main__":  # pragma: no cover
    # `python -m stemify.training.train` predates the console entry point and is in
    # docs and shell histories, so it keeps working -- `main` just lives in `cli` now.
    from stemify.training.cli import main

    main()
