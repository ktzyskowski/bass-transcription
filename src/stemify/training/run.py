"""``RunConfig``: what to run, beside ``TrainConfig``'s how to fit it -- and the
builder that refuses an incoherent pair before any work starts.

A run's settings used to live in two containers with no principle dividing them.
``TrainConfig`` held seventeen fields and ``train()`` took seventeen more parameters,
both filled from the same argparse, and which side a flag landed on was historical:
``--batch-size`` and ``--patience`` were config, ``--limit`` and ``--val-limit`` were
parameters, ``--monitor`` was config while ``--val-every`` was a parameter -- even
though the two have to agree.

The split that does mean something is **what to run** against **how to fit**:

    RunConfig    the corpus, the recipe, the artefacts, the validation cadence, the
                 tracking destination -- everything that identifies *this* run
    TrainConfig  batch size, learning rate, patience, clipping, accumulation --
                 everything the loop would need whatever it was training on

``RunConfig`` holds the ``TrainConfig``, so ``train()`` takes one argument. That is
not cosmetic: the checks below span the two, and while they were a wall of ``if``s at
the top of ``train()`` there was nowhere for them to live that was not "the procedure
that happens to run first".

**Why a builder rather than a dataclass with ``__post_init__``.** Half of these
constraints are about a single field and can be refused the moment it is named --
``--save-epochs`` without ``--out``, a ``--val-every`` of zero, a warm-start path that
is not there. A setter is the tightest place for those: the error arrives at the line
that caused it, and the check sits next to the field it is about instead of in a
shared block that has to re-derive which flags were even passed.

**And why ``build()`` still validates.** Four constraints are irreducibly about *two*
fields, and a setter can only see what has been set already -- with no guaranteed
order, a setter that validated a pair would pass or fail depending on which was named
first. Those live in ``build()``, which is the first moment the whole thing exists:

  - the default val conditions follow the recipe's channel width
  - every val condition must *match* that width
  - a harness ``--monitor`` forces ``--val-every 1``
  - a harness ``--monitor`` must name a metric this run will actually produce

All of it runs before the corpus is indexed, before the CQT cache is touched and
before the first batch -- which is the point. The failure these replace surfaced at
the end of epoch 0.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self

from stemify.conditions import CONDITIONS, canonical
from stemify.fitting.base import Recipe
from stemify.fitting.config import TrainConfig
from stemify.training.validation import harness_metric_names

DEFAULT_EXPERIMENT = "onset-detection"


@dataclass(frozen=True)
class RunConfig:
    """One training run, fully specified. Build it with ``RunBuilder``.

    Frozen, and built through the builder rather than constructed directly, because a
    coherent instance is the thing the checks guarantee -- a field reassigned after
    validation would be a run whose settings were checked in a state it never ran in.
    """

    root: str
    cache_dir: Path
    recipe: Recipe
    fit: TrainConfig

    val_conditions: list[str] = field(default_factory=list)
    val_limit: int = 40
    val_every: int = 5

    limit: int | None = None

    out: Path | None = None
    save_epochs: bool = False
    force: bool = False
    init_from: Path | None = None

    tracking_uri: str | None = None
    experiment: str = DEFAULT_EXPERIMENT
    run_name: str | None = None
    batch_log_every: int = 100

    @property
    def sync_scoring(self) -> bool:
        """Whether the note harness has to run on the critical path.

        A harness metric (``note/*``, ``cues/*``) can only drive checkpoint selection
        if it is scored synchronously: the async validator delivers epoch N's score
        partway through epoch N+1, by which time N's decision is made. So the monitor
        key decides which way the harness runs, rather than being a second flag that
        can disagree with it.
        """
        return not self.fit.monitor.startswith("loss/")

    @property
    def sources(self) -> list[str]:
        """The training conditions, by tag -- what the recipe was built for."""
        return [c.name for c in self.recipe.conditions]


class RunBuilder:
    """Assembles a ``RunConfig``, refusing each setting as it is named.

    Every setter returns ``self``, so a caller reads as a description of the run::

        run = (
            RunBuilder(args.root, args.cache_dir)
            .with_recipe(recipe)
            .fitting(config)
            .validating(conditions, limit=40, every=1)
            .checkpointing(out, save_epochs=True)
            .tracking(uri, experiment, run_name)
            .build()
        )
    """

    def __init__(self, root: str, cache_dir: Path) -> None:
        self._root = root
        self._cache_dir = Path(cache_dir)
        self._recipe: Recipe | None = None
        self._fit: TrainConfig | None = None
        self._val_conditions: list[str] | None = None
        self._val_limit = 40
        self._val_every = 5
        self._limit: int | None = None
        self._out: Path | None = None
        self._save_epochs = False
        self._force = False
        self._init_from: Path | None = None
        self._tracking_uri: str | None = None
        self._experiment = DEFAULT_EXPERIMENT
        self._run_name: str | None = None
        self._batch_log_every = 100

    # -- what is being trained ---------------------------------------------------

    def with_recipe(self, recipe: Recipe) -> Self:
        """The paradigm: batch shape, loss, decoder, and what a checkpoint must hold."""
        self._recipe = recipe
        return self

    def fitting(self, config: TrainConfig) -> Self:
        """The loop's knobs. Kept a separate object rather than flattened in: nothing
        here is specific to this corpus or this experiment, and a second entry point
        that trained on something else would want exactly these and none of the rest."""
        self._fit = config
        return self

    def limited_to(self, tracks: int | None) -> Self:
        """Cap the training set, for a smoke run.

        A limit is a *prefix* of the corpus, so it is one arbitrary slice rather than a
        sample -- see ``corpus.loader``. Zero is refused because it is never what
        anyone means: it trains on nothing and reports a finished run.
        """
        if tracks is not None and tracks < 1:
            raise ValueError(
                f"--limit {tracks} trains on no tracks at all. Omit it to train on the "
                "whole corpus."
            )
        self._limit = tracks
        return self

    # -- validation --------------------------------------------------------------

    def validating(
        self,
        conditions: list[str] | None = None,
        limit: int = 40,
        every: int = 5,
    ) -> Self:
        """Which conditions to score, how many tracks, and how often.

        ``conditions`` of ``None`` defers to the recipe's channel width, which is not
        known until ``build``. Tags go through ``canonical`` here for the same reason
        ``--sources`` does: a run given a retired tag otherwise resolves its *training*
        conditions and then dies looking up its validation ones, minutes later.
        """
        if limit < 1:
            raise ValueError(
                f"--val-limit {limit} scores no tracks, so every harness metric would "
                "be undefined and checkpoint selection would have nothing to read."
            )
        if every < 1:
            raise ValueError(f"--val-every {every} never validates; the minimum is 1.")
        if conditions is not None:
            conditions = [canonical(t) for t in conditions]
            unknown = [t for t in conditions if t not in CONDITIONS]
            if unknown:
                raise ValueError(
                    f"unknown val condition(s) {unknown}; have {sorted(CONDITIONS)}"
                )
        self._val_conditions = conditions
        self._val_limit = limit
        self._val_every = every
        return self

    # -- artefacts ---------------------------------------------------------------

    def checkpointing(
        self,
        out: Path | None = None,
        save_epochs: bool = False,
        force: bool = False,
    ) -> Self:
        """Where this run's checkpoint goes, if anywhere.

        **There is no default path, and the collision check runs here.** There was a
        default (``<store-dir>/onset_model.pt``), and it was wrong twice over: wrong as
        a *name*, because it stopped describing the model when a voicing head arrived
        and stopped describing it three ways over once pitch did; and wrong as a
        *default*, because every run launched without ``--out`` wrote the same file --
        tcn-v1 was overwritten by tcn-v2, a warning went into docs/experiments.md, and
        tcn-v2 was then overwritten by a two-track smoke run anyway.

        So ``out=None`` means *do not save*, which is what a smoke test actually wants,
        and naming a run's checkpoint is a deliberate act. The existence check is here
        rather than at save time because a run that discovers its path is occupied
        after hours of training can only clobber or be thrown away.
        """
        if out is not None and out.exists() and not force:
            raise FileExistsError(
                f"{out} already exists. Name this run's own checkpoint, or pass "
                "--force to overwrite. A checkpoint is the only artefact a result "
                "cannot be reproduced without, so overwriting one is never implicit."
            )
        if save_epochs and out is None:
            raise ValueError(
                "--save-epochs needs --out: it writes '<stem>.epoch<N><suffix>' beside "
                "the run's checkpoint, and there is nowhere to put those without one."
            )
        self._out = out
        self._save_epochs = save_epochs
        self._force = force
        return self

    def warm_starting_from(self, checkpoint: Path | None) -> Self:
        """Initialise the encoder from an existing checkpoint.

        Checked for existence here rather than at load time: a warm start is usually
        the whole point of the run (a semi-CRF starting from good cues rather than
        learning the encoder and the potentials at once), so a typo'd path should not
        quietly become a from-scratch run discovered hours later.
        """
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(
                f"--init-from {checkpoint} does not exist. Without it this would train "
                "from scratch, which is a different experiment."
            )
        self._init_from = checkpoint
        return self

    # -- tracking ----------------------------------------------------------------

    def tracking(
        self,
        uri: str | None = None,
        experiment: str = DEFAULT_EXPERIMENT,
        run_name: str | None = None,
        batch_log_every: int = 100,
    ) -> Self:
        """Where the run records itself. ``uri=None`` records nothing.

        Left permissive on purpose -- a smoke run legitimately wants no tracking -- but
        it is the setting most often forgotten, and a run without it trains, scores and
        checkpoints while recording nothing. ``train()`` logs a warning when it is
        absent rather than refusing, since refusing would break the smoke path.
        """
        if batch_log_every < 1:
            raise ValueError(
                f"--batch-log-every {batch_log_every} would log every batch boundary "
                "as a division by zero; the minimum is 1."
            )
        self._tracking_uri = uri
        self._experiment = experiment
        self._run_name = run_name
        self._batch_log_every = batch_log_every
        return self

    # -- the pair-wise checks ----------------------------------------------------

    def build(self) -> RunConfig:
        """The validated run. Everything here needs *two* fields at once."""
        if self._recipe is None:
            raise ValueError("no recipe: call with_recipe() before build().")
        if self._fit is None:
            raise ValueError("no TrainConfig: call fitting() before build().")

        run = RunConfig(
            root=self._root,
            cache_dir=self._cache_dir,
            recipe=self._recipe,
            fit=self._fit,
            val_conditions=self._default_val_conditions(self._recipe),
            val_limit=self._val_limit,
            val_every=self._val_every,
            limit=self._limit,
            out=self._out,
            save_epochs=self._save_epochs,
            force=self._force,
            init_from=self._init_from,
            tracking_uri=self._tracking_uri,
            experiment=self._experiment,
            run_name=self._run_name,
            batch_log_every=self._batch_log_every,
        )
        _check_channels_match(run)
        _check_monitor_is_reachable(run)
        return run

    def _default_val_conditions(self, recipe: Recipe) -> list[str]:
        """Validate on every deployment condition of the model's own channel width.

        ``mixture`` is deliberately *not* in the default. It is the un-separated
        negative control, and it has been settled since July -- bass buried in a full
        mix scores 0.18-0.27 whatever the model, which is the finding that made
        separation non-negotiable. Scoring it every epoch costs a third of validation
        to re-derive a conclusion no run has moved. It is still one flag away
        (``--val-sources clean,mixture,separated``) for when the question is reopened.
        """
        if self._val_conditions is not None:
            return self._val_conditions
        return (
            ["bichannel-clean", "bichannel-separated"] if recipe.in_channels == 2
            else ["clean", "separated"]
        )


def _check_channels_match(run: RunConfig) -> None:
    mismatched = [
        t for t in run.val_conditions
        if CONDITIONS[t].in_channels != run.recipe.in_channels
    ]
    if mismatched:
        raise ValueError(
            f"val conditions {mismatched} do not match the model's "
            f"{run.recipe.in_channels} input channel(s) (training on {run.sources}). "
            "Scoring needs the same channel layout the model was trained with."
        )


def _check_monitor_is_reachable(run: RunConfig) -> None:
    """A harness ``--monitor`` has to name a metric this run will actually produce,
    on an epoch it will actually be scored.

    Both failures used to surface at the *end of epoch 0*, past the val CQT caching and
    a full pass over the corpus, and only because ``Trainer`` raised on a dict lookup.
    The names are knowable from the metric result types with nothing scored, so there
    is no reason to find out later than here.
    """
    if not run.sync_scoring:
        return
    if run.val_every != 1:
        raise ValueError(
            f"--monitor {run.fit.monitor!r} is scored by the harness, so it must run "
            f"every epoch, but --val-every is {run.val_every}. On a skipped epoch the "
            "monitored metric would not exist and the run would die mid-training."
        )
    known = harness_metric_names(run.val_conditions)
    if run.fit.monitor not in known:
        raise ValueError(
            f"--monitor {run.fit.monitor!r} is not a metric this run will produce. "
            f"Available with --val-sources {run.val_conditions}: {sorted(known)}, "
            "or a loss/<head>/val key."
        )


def with_fit(run: RunConfig, **changes: object) -> RunConfig:
    """A copy of ``run`` with some ``TrainConfig`` fields changed, re-validated.

    The one supported way to alter a built run -- a test that wants a different
    ``monitor`` should get the same checks a command line would.
    """
    return _revalidated(replace(run, fit=replace(run.fit, **changes)))  # type: ignore[arg-type]


def _revalidated(run: RunConfig) -> RunConfig:
    _check_channels_match(run)
    _check_monitor_is_reachable(run)
    return run
