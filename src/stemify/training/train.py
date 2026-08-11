"""Train the frame heads on Slakh, scored by note-level F1.

    uv run stemify-train --root data/raw/slakh2100 \
        --store-dir data/store --tracking-uri file:./mlruns \
        --sources clean,separated --pitch

(``python -m stemify.training.train`` still works; the console script is declared in
``pyproject.toml`` so the entry point is discoverable without reading this file.)

Trains onset and voicing, plus the pitch contour with ``--pitch``. Without it the
model has no pitch source at all and can only be scored at cues level -- pyin, the
fallback that used to make a headless run decodable, is gone.

Domain randomisation: the input is drawn from several *conditions* (``--sources``,
tags from ``core.conditions``) while the target is always the clean-stem MIDI, so
the heads are robust to the condition deployment actually sees. ``clean`` needs no
separation; ``separated`` runs demucs over the train set first, cached.

Validation runs on every deployment condition of the model's own channel width
(``--val-sources``), not just clean stems. Cheap per-frame val loss drives early
stopping inline, while ``AsyncValidationCallback`` scores each checkpoint through the
real ``LearnedEncoder`` + ``GreedyDecoder`` pair -- one ``CheckpointValidator`` per
condition, each caching its CQT so a score is a model forward. From that one forward it
logs both altitudes to MLflow --
``note/<condition>/f1`` (decoder) and ``cues/<condition>/{onset_ap, voicing_auroc}``
(encoder) -- and attaches a per-epoch ``validation/`` JSON artifact with the full
breakdown. The gap between ``note/clean/f1`` and ``note/separated/f1`` is the verdict
on how well the learned cues survive separation.
"""

import logging
from datetime import datetime, timezone

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
from stemify.fitting.base import epoch_batches
from stemify.training.recipes import HEADS
from stemify.training.trainer import Trainer
from stemify.training.run import RunConfig
from stemify.training.validation import build_eval_fn, build_validators

logger = logging.getLogger(__name__)




def train(run: RunConfig) -> None:
    """Fit ``run``'s recipe and, if it names an ``--out``, persist the result.

    One argument. This used to take seventeen, beside a ``TrainConfig`` holding
    seventeen more, and the first 140 lines were the checks that had to run before any
    of it could be trusted -- defaults derived from the recipe, tags canonicalised,
    channel widths matched, the monitor reconciled against the validation cadence.
    All of that is ``RunBuilder``'s now, so what is left here is the procedure: build
    the loaders, build the model, wire the callbacks, fit, save.
    """
    recipe, config = run.recipe, run.fit
    store = Store(run.cache_dir)

    # One store for every condition and both splits. Features are content-addressed on
    # the resolved audio path, so a stem, a mix and a separated mix of the same track
    # land on distinct keys without any tag/split subdirectory to keep them apart --
    # and the val conditions transparently reuse whatever this loop already built.
    # The recipe owns the batch shape: fixed windows for frame BCE, whole tracks for
    # the semi-CRF, whose objective scans a whole segmentation.
    train_loader = recipe.train_loader(run.root, store, config, run.limit)
    val_loader = recipe.val_loader(run.root, store, config, run.val_limit)

    model = recipe.build_model()
    if run.init_from is not None:
        # Encoder weights only. Every recipe builds the same backbone + heads
        # (SemiCRFRecipe.build_model defers to FrameRecipe's), so a checkpoint from
        # one objective initialises another -- which is how a semi-CRF run gets to
        # start from *good* cues instead of learning the encoder and the potentials
        # simultaneously from scratch. Strict: a shape mismatch should fail loudly,
        # not silently leave half the encoder random.
        warm = Checkpoint.load(run.init_from)
        model.load_state_dict(warm.model_state)
        if warm.objective_state:
            logger.warning(
                "%s carries objective parameters (%s); --init-from restores the "
                "encoder only, so these are dropped",
                run.init_from, sorted(warm.objective_state),
            )
        logger.info("initialised encoder from %s", run.init_from)

    trainer = Trainer(model, recipe.build_loss(), config)

    # Written on every improvement, not once at the end: `fit` restores the best
    # weights when it *returns*, so a Ctrl-C, a SIGKILL or an OOM loses the whole run.
    callbacks: list[Callback] = [ConsoleCallback()]
    if run.out is not None:
        callbacks.append(BestCheckpointCallback(run.out, recipe))
        if run.save_epochs:
            callbacks.append(EveryEpochCheckpointCallback(run.out, recipe))
    mlflow = None
    eval_fn = None
    if run.tracking_uri:
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
        name = run.run_name or "-".join(
            [recipe.name, *sorted(set(recipe.heads) - set(HEADS)),
             "".join(sorted(c.name[0] for c in recipe.conditions))]
        )
        mlflow = MLflowCallback(
            run_name=f"{name}-{stamp}", experiment=run.experiment,
            tracking_uri=run.tracking_uri, batch_log_every=run.batch_log_every,
            # Loop-level settings here; strategy knobs come from ``recipe.params``,
            # which is the only thing that knows which of them this run even has.
            params={"encoder": "tcn", "sources": ",".join(run.sources),
                    "val_conditions": ",".join(run.val_conditions),
                    "lr": config.learning_rate,
                    "batch_size": config.batch_size,
                    "accumulate_steps": config.accumulate_steps,
                    "max_grad_norm": config.max_grad_norm,
                    "objective": recipe.name,
                    "in_channels": recipe.in_channels,
                    "train_batches": epoch_batches(train_loader),
                    **recipe.params},
        )
        # Precompute each val condition's CQT (and, for demucs, the separation) once;
        # async scores are then just model forwards. Slowest at construction is demucs,
        # which separates its val audio first.
        logger.info(
            "caching val features for %d tracks x conditions %s...",
            run.val_limit, run.val_conditions,
        )
        validators = build_validators(run.root, run.val_limit, run.val_conditions, store=store)
        # Per-epoch validation reports (all conditions x note+cues) are written here
        # and attached to the run under "validation/", so each scored checkpoint leaves
        # a diffable JSON artifact, not only metric points.
        eval_fn = build_eval_fn(
            validators,
            report_dir=run.cache_dir / "val_reports",
            log_artifact=lambda p, _e: mlflow.log_artifact(p, artifact_path="validation"),
            # Without these a semi-CRF run would be scored as a freshly-initialised
            # GreedyDecoder over its learned cues: plausible numbers, wrong model.
            build_decoder=recipe.decoder_from,
            build_model=recipe.build_model,
        )
        callbacks.append(mlflow)
        if not run.sync_scoring:
            # Off the critical path, delivering to its own sink. Only sound when the
            # score is not what picks the checkpoint.
            callbacks.append(
                AsyncValidationCallback(eval_fn, mlflow.sink, every=run.val_every)
            )

    if run.sync_scoring and eval_fn is None:
        # Reachable: --monitor names a harness metric on a run with no tracking URI, so
        # nothing built the validators. Failing here beats failing at the first epoch
        # boundary an hour in.
        logger.info(
            "caching val features for %d tracks x conditions %s (for --monitor %s)...",
            run.val_limit, run.val_conditions, config.monitor,
        )
        eval_fn = build_eval_fn(
            build_validators(run.root, run.val_limit, run.val_conditions, store=store),
            build_decoder=recipe.decoder_from,
            build_model=recipe.build_model,
        )

    trainer.fit(
        train_loader, val_loader, callbacks=callbacks,
        score_fn=eval_fn if run.sync_scoring else None,
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
    if run.out is None:
        logger.info("no --out given: trained weights are not persisted")
        return
    Checkpoint(
        recipe=recipe,
        model_state=trainer.model.state_dict(),
        objective_state=trainer.loss.state_dict(),
    ).save(run.out)
    logger.info("saved checkpoint -> %s", run.out)
    if mlflow is not None:
        mlflow.log_artifact(run.out)


if __name__ == "__main__":  # pragma: no cover
    # `python -m stemify.training.train` predates the console entry point and is in
    # docs and shell histories, so it keeps working -- `main` just lives in `cli` now.
    from stemify.training.cli import main

    main()
