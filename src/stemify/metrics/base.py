"""``MetricResult``: the one data type every metric returns, frame or note.

A metric is a pure function from a (reference, estimate) pair to numbers. There
are two families, split by the shape of that pair, and each has its own base
class:

  - ``FrameMetric`` (``metrics.frame``) scores two frame-aligned arrays. Its caller
    flattens every track's frames into one pair -- frame scoring is elementwise, so
    concatenation is lossless -- and scores once.
  - ``NoteMetric`` (``metrics.note``) scores two *lists of per-track* note-event
    lists. Note matching is a per-track set assignment, not elementwise, so the
    track structure is load-bearing and cannot be flattened away; the metric
    loops tracks and sums counts internally, then scores once.

What they share is this result type.

**The fields are the schema.** A result used to be a ``name`` plus a bare
``dict[str, float]``, built by one method and read by string literal in the next --
so ``render`` could ask for a key ``compute`` had stopped writing and nothing but a
runtime ``KeyError`` said so. Worse, the key *set* varied with the data: the
degenerate branches of ``OnsetPR`` and ``VoicingAUROC`` returned a single ``nan``
entry and dropped the rest, and those keys are fanned straight into MLflow metric
names by the training callback, so a val pool with no rests silently punched a hole
in a series rather than logging ``nan``.

So a result is now a frozen dataclass whose fields *are* its numbers, and
``SCALARS`` names -- in render order -- the ones that make up its public record.
Two consequences are the point of the exercise:

  - ``render`` reads ``self.rpa``, which mypy checks, rather than ``s["rpa"]``,
    which nothing does.
  - ``SCALARS`` is a ``ClassVar``, so the full set of names a run will log is
    knowable **from the types alone**, with nothing executed. That is what lets
    ``tests/metrics/test_logged_schema.py`` pin every MLflow metric name, and lets
    ``--monitor`` be validated at argument-parse time instead of at the end of
    epoch 0.

``SCALARS`` is a hand-written tuple rather than ``dataclasses.fields`` because two
things it must cover are not fields: *derived* quantities (``octave_gap`` is
``rca - rpa``; P/R/F are functions of the counts) and *order*, which ``render`` and
every report depend on. It is checked at class-creation time, so a name that
resolves to nothing is a ``TypeError`` on import rather than a missing metric
discovered after an epoch of training.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """What a metric computed: named scalars, plus any non-scalar artifacts.

    ``scalars`` is what gets printed, logged, and compared across runs; it is
    derived from ``SCALARS`` so its key set depends on the *type*, never on the
    data. ``arrays`` carries anything that is not a single number -- a confusion
    matrix, a PR curve -- for plotting or further analysis, and subclasses that
    have such artifacts override it. Keeping them apart means a caller can always
    treat ``scalars`` as a flat comparable record regardless of which metric
    produced it.
    """

    name: ClassVar[str]
    """Stable identity, used as the key a report and the training callback file this
    result under. A ``ClassVar`` rather than a field: it is a property of the metric,
    not of one scoring, and making it data meant every result carried a copy."""

    SCALARS: ClassVar[tuple[str, ...]] = ()
    """The public record: which numbers this result reports, in render order.

    Every name must resolve on the class -- as a field or as a property -- which is
    checked when the subclass is created.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a result type that does not declare a resolvable record.

        At class creation, not at first use: an unlogged metric is invisible at the
        point it matters (a run finishes, and the number is simply absent from
        MLflow), so the failure has to happen on import. Runs before ``@dataclass``
        is applied, hence the annotation walk rather than ``fields()``.
        """
        super().__init_subclass__(**kwargs)
        if not cls.SCALARS:
            raise TypeError(
                f"{cls.__name__} declares no SCALARS. A metric result must name the "
                "numbers it reports -- they become MLflow metric names and report keys."
            )
        annotated = {
            key for base in cls.__mro__ for key in getattr(base, "__annotations__", {})
        }
        missing = [k for k in cls.SCALARS if k not in annotated and not hasattr(cls, k)]
        if missing:
            raise TypeError(
                f"{cls.__name__}.SCALARS names {missing}, which is neither a field nor "
                "a property. Every scalar must be readable off the result."
            )

    @property
    def scalars(self) -> dict[str, float]:
        """The public record as a flat mapping, in ``SCALARS`` order.

        Key set fixed by the type, so a caller iterating it into MLflow metric names
        gets the same names every epoch regardless of what the data did.
        """
        return {key: float(getattr(self, key)) for key in self.SCALARS}

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Non-scalar artifacts, for plotting and further analysis. Empty by default."""
        return {}

    def render(self) -> str:
        """Human-readable summary. Subclasses override for domain formatting."""
        return "\n".join(f"  {k:<28} {v:>9.4g}" for k, v in self.scalars.items())
