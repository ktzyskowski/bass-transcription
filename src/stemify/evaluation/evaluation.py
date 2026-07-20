"""Bind a dataset, a baseline and some metrics together, and run them.

    Evaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()

Everything dataset-specific lives behind ``Dataset``; everything metric-specific
lives behind ``Metric``. This module only sequences them:

    for each item:  reference -> predict -> align
    pool all frames
    score overall, then per group, then per item

A baseline that only implements ``predict_pitch`` is not a special case here.
Its single answer becomes a contour held across the reference's voiced frames,
so it flows through the same alignment and the same metrics as a frame-level
tracker -- which is what lets note-level and frame-level baselines appear in one
comparison table.
"""

import sys
from collections.abc import Iterable

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.alignment import FrameSet, ItemRef, align
from stemify.evaluation.metrics import Metric, PitchAccuracy, PitchConfusion
from stemify.evaluation.report import Report

DEFAULT_METRICS: tuple[Metric, ...] = (PitchAccuracy(), PitchConfusion())


class Evaluation:
    """One baseline, one dataset, one set of metrics."""

    def __init__(
        self,
        dataset: Dataset,
        baseline: PitchBaseline,
        metrics: Iterable[Metric] = DEFAULT_METRICS,
        group_by: Iterable[str] | None = None,
        progress: bool = True,
    ) -> None:
        self.dataset = dataset
        self.baseline = baseline
        self.metrics = list(metrics)
        # None means "whatever the items declare"; an explicit list narrows it.
        self.group_by = list(group_by) if group_by is not None else None
        self.progress = progress

    def _estimate(self, item: EvalItem, reference: Contour) -> Contour:
        """The baseline's answer, as a contour either way."""
        path = item.audio()
        if self.baseline.is_frame_level:
            return self.baseline.predict_f0(path)

        midi = self.baseline.predict_pitch(path)
        if midi is None:
            return Contour(reference.times, np.zeros_like(reference.freqs))
        return Contour.constant(
            librosa.midi_to_hz(midi), reference.times, reference.voiced
        )

    def collect(self) -> FrameSet:
        """Predict and align every item, pooling the result."""
        parts: list[tuple[ItemRef, np.ndarray, np.ndarray]] = []
        total = len(self.dataset)
        for i, item in enumerate(self.dataset, 1):
            reference = item.reference()
            # Nothing sounds, so there is no pitch question to score. Skipping
            # keeps these out of the denominator rather than counting them as
            # failures the baseline never had a chance at.
            if reference.is_silent:
                continue
            estimate = self._estimate(item, reference)
            true, pred = align(reference, estimate)
            parts.append((ItemRef(item.name, item.groups), true, pred))
            if self.progress:
                print(f"  {i}/{total} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)
        return FrameSet.from_items(parts)

    def run(self) -> Report:
        frames = self.collect()

        # Refuse to build a report for an empty run. Every rate would come out
        # 0.0%, which reads as "the baseline scored zero" rather than "nothing
        # was evaluated" -- indistinguishable in the output, and only one of
        # them is a result.
        if not frames.items:
            raise ValueError(
                f"No items to evaluate for {self.dataset.name}. Check the dataset "
                "root, and whether filters (polyphony, split, limit) excluded "
                "everything."
            )
        if not frames.voiced.any():
            raise ValueError(
                f"{self.dataset.name}: {len(frames.items)} items but no "
                "reference-voiced frames. The reference contours are all silent."
            )

        keys = self.group_by if self.group_by is not None else frames.group_keys()
        breakdowns = {}
        for key in keys:
            buckets = {
                value: [m.calculate(sub.true, sub.pred) for m in self.metrics]
                for value, sub in frames.by(key).items()
                if sub.voiced.any()
            }
            # A key every item shares says nothing -- one bucket is the aggregate
            # again, printed under a heading that implies a comparison.
            if len(buckets) > 1:
                breakdowns[key] = buckets

        accuracy = PitchAccuracy()
        per_item = []
        for i, ref in enumerate(frames.items):
            sub = frames.where(frames.item == i)
            if sub.voiced.any():
                per_item.append((ref.name, accuracy.calculate(sub.true, sub.pred).scalars))

        return Report(
            baseline=self.baseline.name,
            dataset=self.dataset.name,
            n_items=len(frames.items),
            n_frames=len(frames),
            overall=[m.calculate(frames.true, frames.pred) for m in self.metrics],
            breakdowns=breakdowns,
            per_item=per_item,
        )
