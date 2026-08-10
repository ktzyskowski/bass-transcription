"""The frame-metric contracts: aligned arrays, and the two things they can hold.

A frame metric receives two frame-aligned arrays and returns a ``MetricResult``. It
does no resampling, no voicing inference, no aggregation -- ``CuesEvaluation`` has
already done all of that (see ``stemify.evaluation.cues``). That is the whole point of
the split: metrics stay stateless and comparable, and alignment is implemented once
rather than once per metric.

**There are two contracts here, not one loose one.** ``FrameMetric`` used to declare
that values were "frequencies in Hz, with 0.0 meaning unvoiced" -- which is true of
half its subclasses. ``OnsetPR`` and ``VoicingAUROC`` take a label and a probability,
and both carried a paragraph in their own docstring re-negotiating the contract above
them (*"the FrameMetric contract is only that the two arrays are frame-aligned"*).
That sentence described ``_check``, not the docstring it was arguing with. So the
shared base now promises only what it enforces, and the two real contracts are
subclasses that each add their own:

    FrameMetric      1-D, equal length, a common time grid
    ├── PitchMetric      + both arrays are Hz, with 0.0 meaning unvoiced
    └── RankingMetric    + a target and a score, both in [0, 1]

``RankingMetric``'s check is not bookkeeping. ``Frames`` documents ``PROBABILITY_HEADS``
as a property of its own contract, but enforces only shape in ``__post_init__`` -- the
sigmoid that would make it true lives a layer away in ``models.frames``. Handing raw
logits to ``VoicingAUROC`` yields a *perfectly plausible* AUROC, because AUROC is
rank-based and invariant to any monotone squash; what breaks silently is
``best_threshold``, which would then be a cut on the logit scale reported to a decoder
that thresholds probabilities.

Aggregation across a dataset is *not* a frame metric's concern. ``CuesEvaluation``
pools frames from every track into one pair of arrays and calls ``compute``
once, which makes results frame-weighted by construction. There is deliberately no
``combine`` step to get wrong -- which is exactly where the note-level family
diverges: ``NoteMetric`` (``metrics.note``) *cannot* pool its inputs, so it scores per
item and sums, and needs a combine step this one does not.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.metrics import precision_recall_curve

from stemify.metrics.base import MetricResult


class FrameMetric(ABC):
    """Computes one family of numbers from an aligned (reference, estimate) pair.

    The base contract is alignment and nothing more: two 1-D arrays of equal length
    on a common time grid, the grid being the *reference's* (the mir_eval convention
    -- the reference defines where questions get asked, the estimate is resampled
    onto it). What the values mean is the subclass's contract.
    """

    name: ClassVar[str]

    @abstractmethod
    def compute(self, true: np.ndarray, pred: np.ndarray) -> MetricResult:
        """Score aligned reference vs. estimate. See the class for the contract both
        arrays must satisfy."""

    def _check(self, true: np.ndarray, pred: np.ndarray) -> None:
        """Enforce the alignment contract. Cheap, and the failure it catches --
        silently comparing misaligned grids -- produces plausible-looking
        numbers that are entirely meaningless."""
        if true.shape != pred.shape:
            raise ValueError(
                f"{self.name}: expected aligned arrays, got {true.shape} vs "
                f"{pred.shape}. Align with stemify.evaluation.alignment first."
            )
        if true.ndim != 1:
            raise ValueError(f"{self.name}: expected 1-D arrays, got {true.ndim}-D")


class PitchMetric(FrameMetric):
    """A frame metric over two Hz arrays, where **0.0 means unvoiced**.

    The convention comes from mir_eval's melody protocol and is what lets one array
    carry both "what pitch" and "anything at all": there is no separate voicing mask
    to keep in step with the frequencies.
    """

    def _check(self, true: np.ndarray, pred: np.ndarray) -> None:
        super()._check(true, pred)
        # A negative frequency is not a quieter one -- it is a sign that something
        # upstream is in cents, in MIDI, or in a log domain, all of which produce
        # believable-looking accuracy against a reference read the same way.
        for label, values in (("reference", true), ("estimate", pred)):
            if values.size and values.min() < 0.0:
                raise ValueError(
                    f"{self.name}: {label} has negative values (min "
                    f"{values.min():.4g}). These arrays are frequencies in Hz with "
                    "0.0 meaning unvoiced, not cents or MIDI numbers."
                )


class RankingMetric(FrameMetric):
    """A frame metric over a per-frame target and a per-frame score, both in [0, 1].

    ``true`` is the target (1 on the positive class, 0 elsewhere) and ``pred`` is the
    head's probability. Threshold-free by design: turning a curve into decisions is a
    decoder's choice, and mixing it in conflates *how good is the head* with *where
    should the cut be*.
    """

    def _check(self, true: np.ndarray, pred: np.ndarray) -> None:
        super()._check(true, pred)
        for label, values in (("target", true), ("score", pred)):
            if values.size and (values.min() < 0.0 or values.max() > 1.0):
                raise ValueError(
                    f"{self.name}: {label} ranges [{values.min():.4g}, "
                    f"{values.max():.4g}], outside [0, 1]. These are a target and a "
                    "probability -- raw logits score a plausible AUROC and a "
                    "meaningless best_threshold."
                )

    @staticmethod
    def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
        """The probability cut that maximises frame-F1, and that F1.

        The head's best operating point, to compare a decoder's default cut against.
        Shared by both ranking metrics: it was duplicated verbatim in each, including
        the ``1e-12`` and the ``[:-1]`` off-by-one guard -- the two parts most likely
        to need a matched fix.
        """
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        if thresholds.size == 0:
            return 0.5, 0.0
        # precision/recall carry one extra entry (recall=0, with no threshold); pair
        # each threshold with its precision/recall before taking the F1-maximising cut.
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best = int(np.argmax(f1))
        return float(thresholds[best]), float(f1[best])
