"""``MetricResult``: the record is a property of the type, not of the data.

The defect this replaces was quiet. A result was a ``name`` plus a bare
``dict[str, float]``, and the degenerate branches of the two threshold-free metrics
returned a one-entry dict -- so a val pool with no onsets, or with no rests, dropped
three keys from that epoch. The training callback fans those keys straight into
MLflow metric names, so the series did not record ``nan``; it simply had a hole, and
nothing raised.

So the property under test is not "the numbers are right" (the per-metric files do
that) but "**the key set does not depend on the input**", plus the class-creation
checks that keep a new result type from reintroducing the same shape.
"""

import math

import numpy as np
import pytest

from stemify.metrics import (
    MetricResult,
    OnsetPR,
    PitchAccuracy,
    PitchConfusion,
    PRF,
    Fragmentation,
    VoicingAUROC,
)

# One entry per metric: the metric, a healthy input, and a degenerate one that used to
# collapse the record. Degenerate is not exotic -- an all-voiced val pool and a track
# with no annotated onsets both occur.
CASES = [
    pytest.param(
        OnsetPR(),
        (np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.8])),
        (np.zeros(4), np.array([0.1, 0.9, 0.2, 0.8])),
        id="onset_pr",
    ),
    pytest.param(
        VoicingAUROC(),
        (np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.8])),
        (np.ones(4), np.array([0.1, 0.9, 0.2, 0.8])),
        id="voicing_auroc",
    ),
    pytest.param(
        PitchAccuracy(),
        (np.array([110.0, 0.0, 220.0]), np.array([110.0, 0.0, 220.0])),
        (np.zeros(3), np.zeros(3)),
        id="pitch_accuracy",
    ),
    pytest.param(
        PitchConfusion(),
        (np.array([110.0, 0.0, 220.0]), np.array([110.0, 0.0, 220.0])),
        (np.zeros(3), np.zeros(3)),
        id="pitch_confusion",
    ),
]


@pytest.mark.parametrize("metric,healthy,degenerate", CASES)
def test_the_key_set_is_the_same_on_healthy_and_degenerate_input(
    metric, healthy, degenerate
):
    """The regression itself. Values may go ``nan``; names may not go missing."""
    assert set(metric.compute(*healthy).scalars) == set(
        metric.compute(*degenerate).scalars
    )


@pytest.mark.parametrize("metric,healthy,degenerate", CASES)
def test_the_key_set_is_exactly_what_the_type_declares(metric, healthy, degenerate):
    """``SCALARS`` is readable off the class, which is what lets a run's full set of
    MLflow metric names be known before anything is scored."""
    for args in (healthy, degenerate):
        result = metric.compute(*args)
        assert tuple(result.scalars) == result.SCALARS


def test_a_degenerate_result_reports_nan_rather_than_omitting_the_number():
    """``nan`` is a fact -- "this could not be computed on this pool" -- and it plots
    as a gap. An absent key is indistinguishable from a metric that was never wired
    up."""
    result = OnsetPR().compute(np.zeros(4), np.array([0.1, 0.9, 0.2, 0.8]))
    assert math.isnan(result.onset_ap)
    assert math.isnan(result.onset_best_f1)
    # Not everything is unknown: the positive rate is 0.0, which is *why* the rest is.
    assert result.onset_positive_rate == 0.0


def test_note_results_declare_their_record_too():
    for metric in (PRF(), Fragmentation()):
        result = metric.compute([[]], [[]])
        assert tuple(result.scalars) == result.SCALARS


# --------------------------------------------------------------------------- #
# The class-creation checks: a new result type cannot reintroduce the shape.  #
# --------------------------------------------------------------------------- #
def test_a_result_without_a_record_is_refused_at_class_creation():
    """On import rather than at first use: an unlogged metric is invisible exactly
    where it matters -- a run finishes and the number is simply not in MLflow."""
    with pytest.raises(TypeError, match="declares no SCALARS"):

        class _Bare(MetricResult):
            name = "bare"


def test_a_record_naming_something_unreadable_is_refused():
    with pytest.raises(TypeError, match="neither a field nor a property"):

        class _Typo(MetricResult):
            name = "typo"
            SCALARS = ("recal",)  # one 'l'

            recall: float


def test_a_derived_property_counts_as_part_of_the_record():
    """Why ``SCALARS`` is written out rather than taken from ``dataclasses.fields``:
    ``octave_gap`` and P/R/F are not fields, and they are exactly the numbers a run is
    read by."""
    result = PitchAccuracy().compute(
        np.array([110.0, 220.0]), np.array([110.0, 440.0])
    )
    assert "octave_gap" in result.scalars
    assert result.scalars["octave_gap"] == pytest.approx(result.rca - result.rpa)
