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
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.alignment import FrameSet, ItemRef, align
from stemify.evaluation.metrics import Metric, PitchAccuracy, PitchConfusion
from stemify.evaluation.report import Report
from stemify.transforms import AudioTransform, Identity

DEFAULT_METRICS: tuple[Metric, ...] = (PitchAccuracy(), PitchConfusion())


def _estimate_contour(
    baseline: PitchBaseline,
    transform: AudioTransform,
    item: EvalItem,
    reference: Contour,
) -> Contour:
    """The baseline's answer for one item, as a contour either way.

    A free function rather than a method so it is reachable from a worker
    process: ``collect`` can hand ``(baseline, transform, item)`` to a pool and
    the worker reconstructs the estimate here.
    """
    path = transform.apply(item.audio())
    if baseline.is_frame_level:
        return baseline.predict_f0(path)
    midi = baseline.predict_pitch(path)
    if midi is None:
        return Contour(reference.times, np.zeros_like(reference.freqs))
    return Contour.constant(librosa.midi_to_hz(midi), reference.times, reference.voiced)


def _evaluate_item(
    payload: tuple[PitchBaseline, AudioTransform, EvalItem],
) -> tuple[ItemRef, np.ndarray, np.ndarray] | None:
    """Score one item to pooled arrays. Returns None for a silent reference.

    The pool's unit of work. Everything it touches -- the baseline, the
    transform, the item, and the returned arrays -- must pickle, which the
    concrete types here do. A GPU transform is the exception: with a cold cache
    each worker would load its own model and contend for one device, so
    parallelism is for the CPU-bound conditions (pyin on stems/mixes) or a
    warm-cache Demucs pass, not cold separation. See ``Evaluation.collect``.
    """
    baseline, transform, item = payload
    reference = item.reference()
    if reference.is_silent:
        return None
    true, pred = align(reference, _estimate_contour(baseline, transform, item, reference))
    return ItemRef(item.name, item.groups), true, pred


class Evaluation:
    """One baseline, one dataset, one set of metrics."""

    def __init__(
        self,
        dataset: Dataset,
        baseline: PitchBaseline,
        metrics: Iterable[Metric] = DEFAULT_METRICS,
        transform: AudioTransform | None = None,
        group_by: Iterable[str] | None = None,
        progress: bool = True,
        workers: int = 1,
    ) -> None:
        self.dataset = dataset
        self.baseline = baseline
        self.metrics = list(metrics)
        # Preprocessing between the dataset's audio and the baseline's ears --
        # source separation, principally. Identity rather than None so every
        # report carries a transform label and the three-condition comparison
        # differs only in configuration.
        self.transform = transform or Identity()
        # None means "whatever the items declare"; an explicit list narrows it.
        self.group_by = list(group_by) if group_by is not None else None
        self.progress = progress
        # Process-pool width for collect(). The per-item work is independent and
        # CPU-bound (pyin runs at ~1x realtime on long tracks, single-threaded),
        # so this is close to linear in core count. Leave at 1 for a GPU
        # transform on a cold cache -- one device does not parallelise across
        # processes; pre-warm the cache instead.
        self.workers = max(1, workers)

    def _estimate(self, item: EvalItem, reference: Contour) -> Contour:
        """The baseline's answer, as a contour either way."""
        return _estimate_contour(self.baseline, self.transform, item, reference)

    def collect(self) -> FrameSet:
        """Predict and align every item, pooling the result."""
        items = list(self.dataset)
        if self.workers > 1:
            return self._collect_parallel(items)

        parts: list[tuple[ItemRef, np.ndarray, np.ndarray]] = []
        for i, item in enumerate(items, 1):
            result = _evaluate_item((self.baseline, self.transform, item))
            # None => silent reference; skip so it stays out of the denominator
            # rather than counting as a failure the baseline never had a shot at.
            if result is not None:
                parts.append(result)
            if self.progress:
                print(f"  {i}/{len(items)} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)
        return FrameSet.from_items(parts)

    def _collect_parallel(self, items: list[EvalItem]) -> FrameSet:
        payloads = [(self.baseline, self.transform, item) for item in items]
        parts: list[tuple[ItemRef, np.ndarray, np.ndarray]] = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            # map keeps submission order, so per-item reporting stays stable
            # across runs regardless of which worker finishes first.
            for i, result in enumerate(pool.map(_evaluate_item, payloads), 1):
                if result is not None:
                    parts.append(result)
                if self.progress:
                    print(f"  {i}/{len(items)}", end="\r", file=sys.stderr)
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
            transform=self.transform.name,
            n_items=len(frames.items),
            n_frames=len(frames),
            overall=[m.calculate(frames.true, frames.pred) for m in self.metrics],
            breakdowns=breakdowns,
            per_item=per_item,
        )
