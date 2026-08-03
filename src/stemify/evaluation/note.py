"""Note-level evaluation: bind a dataset and a ``Segmenter``, and score notes.

    NoteEvaluation(SlakhDataset("data/raw/slakh2100"),
                   Segmenter(ClassicalEncoder(), HeuristicDecoder())).run()

The cues-level harness (``CuesEvaluation``) scores the encoder's per-frame evidence.
This asks "did you find the right notes" -- the metric a tab is actually graded on,
and the one a bass-transcription collaborator's F1 numbers use.

It scores the exact object deployment ships: a ``Segmenter(encoder, decoder)``. The
DSP floor is ``Segmenter(ClassicalEncoder(), HeuristicDecoder())``; a trained model is
``Segmenter(LearnedEncoder(ckpt), decoder)`` -- so eval and deploy transcribe through
one object and cannot drift, and there is no separate "learned baseline" class. (A
note-native transcriber that emits notes without a contour would plug in the same
way once it exposes ``segment(sources) -> notes``.)

Same shape as the cues harness (dataset + object -> a ``Report``), and the same
division of labour: everything dataset-specific lives behind ``Dataset``, everything
metric-specific behind ``NoteMetric``. The one difference is where the per-track step
sits -- note metrics cannot pool their inputs (matching is per track), so this collects
each item's reference and estimate note lists, then hands the whole nested list to each
metric, which loops tracks internally (a frame metric pools frames and scores once).
"""

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.report import Report, jsonable
from stemify.metrics import (
    DEFAULT_NOTE_METRICS,
    MetricResult,
    NoteMetric,
    prf_rates,
    score_notes,
)
from stemify.notes import NoteEvent
from stemify.segmenter import Segmenter
from stemify.sources import Sources
from stemify.transforms import AudioTransform, Identity


def _per_item_prf(
    overall: list[MetricResult], names: list[str]
) -> list[tuple[str, dict[str, float]]]:
    """``(name, PRF scalars)`` per track, projected from PRF's ``per_item_counts``.

    Finds the PRF result among the scored metrics and turns each track's stored
    ``(tp, fp, fn)`` row into P/R/F via the same ``prf_rates`` the pooled totals use --
    so the worst-item ranking and the headline numbers come from one match. Empty when
    no PRF was scored."""
    counts = next(
        (r.arrays["per_item_counts"] for r in overall if "per_item_counts" in r.arrays),
        None,
    )
    if counts is None:
        return []
    return [(name, prf_rates(*(int(c) for c in row))) for name, row in zip(names, counts)]


@dataclass(frozen=True)
class NoteReport(Report):
    """Note-level result: each metric's output, plus per-item PRF for breakdown."""

    transcriber: str
    dataset: str
    n_items: int
    overall: list[MetricResult]
    # How the audio was prepared before the encoder heard it. Defaulted so existing
    # callers keep working, but always recorded: a result is not interpretable
    # without knowing whether it came from a clean stem or a separated one.
    transform: str = "identity"
    # (name, PRF scalars) per item -- for ranking the tracks the transcriber fails
    # outright, which a pooled F1 hides.
    per_item: list[tuple[str, dict[str, float]]] = field(default_factory=list)

    def scalar(self, metric: str, key: str) -> float | None:
        """One number by name, for comparing runs without parsing text."""
        for result in self.overall:
            if result.name == metric:
                return result.scalars.get(key)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcriber": self.transcriber,
            "dataset": self.dataset,
            "transform": self.transform,
            "level": "note",
            "n_items": self.n_items,
            "overall": {
                r.name: {
                    "scalars": {k: jsonable(v) for k, v in r.scalars.items()},
                    "arrays": {k: jsonable(v) for k, v in r.arrays.items()},
                }
                for r in self.overall
            },
            "per_item": [
                {"name": name, **{k: jsonable(v) for k, v in scalars.items()}}
                for name, scalars in self.per_item
            ],
        }

    def render(self, worst: int = 10) -> str:
        via = "" if self.transform == "identity" else f", audio via {self.transform}"
        lines = [
            (
                f"\n=== {self.transcriber} on {self.dataset} "
                f"(note-level, {self.n_items:,} items{via}) ==="
            ),
        ]
        for result in self.overall:
            lines += [f"\n{result.name}", result.render()]

        # A pooled F1 hides a handful of tracks the transcriber fails outright,
        # which is a different problem from being uniformly mediocre.
        if len(self.per_item) > 1 and worst:
            ranked = sorted(self.per_item, key=lambda pair: pair[1]["f1"])[:worst]
            lines.append(f"\nworst {len(ranked)} items by F1")
            lines.append(f"  {'F1':>7} {'tp':>5} {'fp':>5} {'fn':>5}  item")
            for name, s in ranked:
                lines.append(
                    f"  {s['f1']:>7.1%} {int(s['tp']):>5} {int(s['fp']):>5} "
                    f"{int(s['fn']):>5}  {name[:44]}"
                )
        return "\n".join(lines)


class NoteEvaluation:
    """Bind a dataset and a ``Segmenter``; ``run()`` scores its notes.

    ``transform`` preprocesses the audio before the encoder hears it (separation,
    principally); the reference notes are ground truth and untouched by it, so only
    the estimate side sees the transformed audio.
    """

    def __init__(
        self,
        dataset: Dataset,
        transcriber: Segmenter,
        metrics: Iterable[NoteMetric] = DEFAULT_NOTE_METRICS,
        transform: AudioTransform | None = None,
        progress: bool = True,
    ) -> None:
        self.dataset = dataset
        self.transcriber = transcriber
        self.metrics = list(metrics)
        self.transform = transform if transform is not None else Identity()
        self.progress = progress

    def _estimate_notes(self, item: EvalItem) -> list[NoteEvent]:
        path = self.transform.apply(item.audio())
        return self.transcriber.segment(Sources(bass=path))

    def run(self) -> NoteReport:
        items = list(self.dataset)
        names: list[str] = []
        references: list[list[NoteEvent]] = []
        estimates: list[list[NoteEvent]] = []
        for i, item in enumerate(items, 1):
            names.append(item.name)
            references.append(item.reference_notes())
            estimates.append(self._estimate_notes(item))
            if self.progress:
                print(f"  {i}/{len(items)} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)

        if not names:
            raise ValueError(
                f"No items to evaluate for {self.dataset.name}. Check the dataset "
                "root, and whether filters (polyphony, split, limit) excluded "
                "everything."
            )

        overall = score_notes(references, estimates, self.metrics)

        # The worst-item breakdown reads PRF's per-track counts straight off its result
        # -- the same single match the pooled numbers came from, no re-scoring. Empty if
        # the metric list carries no PRF (nothing to rank tracks by).
        per_item = _per_item_prf(overall, names)

        return NoteReport(
            transcriber=self.transcriber.name,
            dataset=self.dataset.name,
            transform=self.transform.name,
            n_items=len(names),
            overall=overall,
            per_item=per_item,
        )
