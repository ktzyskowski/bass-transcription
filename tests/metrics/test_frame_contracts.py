"""The two frame-metric contracts, and what each refuses.

``FrameMetric`` used to declare one contract -- "frequencies in Hz, with 0.0 meaning
unvoiced" -- that half its subclasses did not satisfy, and ``OnsetPR`` and
``VoicingAUROC`` each carried a paragraph arguing with it. That paragraph was right
about ``_check`` and wrong about the docstring above it, which is the signature of a
contract nothing enforces: prose drifts, and only the parts with a raise behind them
stay true.

So there are two contracts now, each with its own guard. What is tested here is that
each one refuses its own violation -- not that the base is documented differently.
"""

import numpy as np
import pytest

from stemify.metrics import OnsetPR, PitchAccuracy, PitchConfusion, VoicingAUROC

RANKING = [pytest.param(OnsetPR(), id="onset"), pytest.param(VoicingAUROC(), id="voicing")]
PITCH = [
    pytest.param(PitchAccuracy(), id="accuracy"),
    pytest.param(PitchConfusion(), id="confusion"),
]
ALL = RANKING + PITCH


# --------------------------------------------------------------------------- #
# Shared: alignment                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ALL)
def test_misaligned_lengths_are_refused(metric):
    """The failure this catches produces plausible numbers rather than an error, and
    a pooled score cannot be sanity-checked by eye."""
    with pytest.raises(ValueError, match="aligned arrays"):
        metric.compute(np.zeros(4), np.zeros(5))


@pytest.mark.parametrize("metric", ALL)
def test_two_dimensional_input_is_refused(metric):
    with pytest.raises(ValueError, match="1-D"):
        metric.compute(np.zeros((2, 3)), np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# RankingMetric: a target and a probability                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", RANKING)
def test_raw_logits_are_refused(metric):
    """The guard that earns its keep. AUROC and average precision are rank-based, so
    logits score *identically* to the probabilities they came from -- the number looks
    fine. What silently breaks is ``best_threshold``, reported to a decoder that cuts
    at 0.5 on a probability.

    ``Frames`` declares ``PROBABILITY_HEADS`` as part of its own contract but checks
    only shape; the sigmoid that would make it true lives in ``models.frames``, a
    layer away. This is the second line of that defence.
    """
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    logits = np.array([-2.0, 3.0, -1.5, 2.5])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        metric.compute(labels, logits)


@pytest.mark.parametrize("metric", RANKING)
def test_a_target_outside_the_unit_interval_is_refused(metric):
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        metric.compute(np.array([0.0, 2.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.8]))


@pytest.mark.parametrize("metric", RANKING)
def test_probabilities_at_the_bounds_are_fine(metric):
    """``ClassicalEncoder`` emits a 0/1 spike train, which is a probability."""
    metric.compute(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# PitchMetric: Hz, with 0.0 meaning unvoiced                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", PITCH)
def test_negative_frequencies_are_refused(metric):
    """A negative value here means the array is in cents, in MIDI, or in a log domain
    -- all of which score believably against a reference read the same way, and
    meaninglessly against one that is not."""
    with pytest.raises(ValueError, match="frequencies in Hz"):
        metric.compute(np.array([110.0, 220.0]), np.array([-1200.0, 220.0]))


@pytest.mark.parametrize("metric", PITCH)
def test_zero_is_unvoiced_not_a_violation(metric):
    metric.compute(np.array([0.0, 220.0]), np.array([0.0, 220.0]))


@pytest.mark.parametrize("metric", PITCH)
def test_a_negative_reference_is_refused_too(metric):
    """Both sides, because the reference is just as capable of arriving in the wrong
    units -- a corpus loader is where that conversion happens."""
    with pytest.raises(ValueError, match="reference"):
        metric.compute(np.array([-110.0, 220.0]), np.array([110.0, 220.0]))
