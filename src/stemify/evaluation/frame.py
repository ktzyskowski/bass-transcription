"""Frame-level evaluation: bind a dataset, a baseline and metrics, and run them.

    FrameEvaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()

Asks "what pitch at each instant". Everything dataset-specific lives behind
``Dataset``, everything metric-specific behind ``FrameMetric``; this module only
sequences them:

    for each item:  reference -> predict -> align
    pool all frames
    score overall, then per group, then per item

A baseline that only implements ``predict_pitch`` is not a special case here.
Its single answer becomes a contour held across the reference's voiced frames,
so it flows through the same alignment and the same metrics as a frame-level
tracker.

The note-level counterpart is ``stemify.evaluation.note`` -- same shape
(dataset + baseline -> a ``Report``), different question ("did you find the right
notes") and a genuinely different data flow (interval matching, not frame
pooling), which is why the two are separate modules rather than one.
"""

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.alignment import FrameSet, ItemRef, align
from stemify.evaluation.report import Report, jsonable
from stemify.metrics import FrameMetric, PitchAccuracy, PitchConfusion
from stemify.metrics.base import MetricResult
from stemify.parallel import parallel_map
from stemify.transforms import AudioTransform, Identity

DEFAULT_METRICS: tuple[FrameMetric, ...] = (PitchAccuracy(), PitchConfusion())


@dataclass(frozen=True)
class FrameReport(Report):
    """Frame-level result: metric outputs overall, by group, and per item."""

    baseline: str
    dataset: str
    n_items: int
    n_frames: int
    overall: list[MetricResult]
    # How the audio was prepared before the baseline heard it. Defaulted so
    # existing callers keep working, but always recorded, because a result is
    # not interpretable without knowing whether it came from a clean stem or a
    # separated one.
    transform: str = "identity"
    breakdowns: dict[str, dict[str, list[MetricResult]]] = field(default_factory=dict)
    per_item: list[tuple[str, dict[str, float]]] = field(default_factory=list)

    def scalar(self, metric: str, key: str) -> float | None:
        """One number by name, for comparing runs without parsing text."""
        for result in self.overall:
            if result.name == metric:
                return result.scalars.get(key)
        return None

    def to_dict(self) -> dict:
        """Serialisable form. Arrays are kept -- a confusion matrix is small and
        is the part you would want to plot later from a stored run."""
        return {
            "baseline": self.baseline,
            "dataset": self.dataset,
            "transform": self.transform,
            "level": "frame",
            "n_items": self.n_items,
            "n_frames": self.n_frames,
            "overall": {
                r.name: {
                    "scalars": {k: jsonable(v) for k, v in r.scalars.items()},
                    "arrays": {k: jsonable(v) for k, v in r.arrays.items()},
                }
                for r in self.overall
            },
            "breakdowns": {
                key: {
                    value: {
                        r.name: {k: jsonable(v) for k, v in r.scalars.items()}
                        for r in results
                    }
                    for value, results in buckets.items()
                }
                for key, buckets in self.breakdowns.items()
            },
            "per_item": [
                {"name": name, **{k: jsonable(v) for k, v in scalars.items()}}
                for name, scalars in self.per_item
            ],
        }

    def render(self, worst: int = 10) -> str:
        lines = [
            f"\n=== {self.baseline} on {self.dataset} "
            f"({self.n_items:,} items, {self.n_frames:,} frames) ===",
        ]
        if self.transform != "identity":
            lines.append(f"    audio preprocessed by: {self.transform}")
        for result in self.overall:
            lines += [f"\n{result.name}", result.render()]

        for key, buckets in self.breakdowns.items():
            lines.append(f"\nby {key}")
            lines.append(f"  {key:<14} {'n':>9} {'RPA':>8} {'RCA':>8} {'gap':>8}")
            for value, results in buckets.items():
                scalars = next(
                    (r.scalars for r in results if r.name == "pitch-accuracy"), None
                )
                if scalars is None:
                    continue
                lines.append(
                    f"  {value[:14]:<14} {scalars['n_voiced']:>9,.0f} "
                    f"{scalars['rpa']:>8.1%} {scalars['rca']:>8.1%} "
                    f"{scalars['octave_gap']:>8.1%}"
                )

        # A good aggregate can hide a handful of items the baseline fails
        # outright, which is a different problem from being uniformly mediocre.
        if len(self.per_item) > 1 and worst:
            ranked = sorted(self.per_item, key=lambda pair: pair[1]["rpa"])[:worst]
            lines.append(f"\nworst {len(ranked)} items by RPA")
            lines.append(f"  {'RPA':>8} {'oct-gap':>8}  item")
            for name, scalars in ranked:
                lines.append(
                    f"  {scalars['rpa']:>8.1%} {scalars['octave_gap']:>8.1%}  {name[:50]}"
                )
        return "\n".join(lines)


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
    warm-cache Demucs pass, not cold separation. See ``FrameEvaluation.collect``.
    """
    baseline, transform, item = payload
    reference = item.reference()
    if reference.is_silent:
        return None
    true, pred = align(reference, _estimate_contour(baseline, transform, item, reference))
    return ItemRef(item.name, item.groups), true, pred


class FrameEvaluation:
    """One baseline, one dataset, one set of metrics, scored per frame."""

    def __init__(
        self,
        dataset: Dataset,
        baseline: PitchBaseline,
        metrics: Iterable[FrameMetric] = DEFAULT_METRICS,
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
        n = len(items)

        def report(i: int, _result: object) -> None:
            # Runs in this process as each result streams back, in submission
            # order, so per-item reporting stays stable across runs.
            if self.progress:
                print(f"  {i}/{n}", end="\r", file=sys.stderr)

        results = parallel_map(
            _evaluate_item, payloads, workers=self.workers, on_result=report
        )
        if self.progress:
            print(file=sys.stderr)
        # None => silent reference; drop so it stays out of the denominator.
        parts = [r for r in results if r is not None]
        return FrameSet.from_items(parts)

    def run(self) -> FrameReport:
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

        return FrameReport(
            baseline=self.baseline.name,
            dataset=self.dataset.name,
            transform=self.transform.name,
            n_items=len(frames.items),
            n_frames=len(frames),
            overall=[m.calculate(frames.true, frames.pred) for m in self.metrics],
            breakdowns=breakdowns,
            per_item=per_item,
        )
