"""Note-level evaluation: bind a dataset, a baseline and note metrics, and run.

    NoteEvaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()

The frame-level harness (``FrameEvaluation``) asks "what pitch at each instant".
This asks "did you find the right notes" -- the metric a tab is actually graded
on, and the one a bass-transcription collaborator's F1 numbers use, so it is
what makes those numbers comparable.

Same shape as the frame harness (dataset + baseline -> a ``Report``), and the
same division of labour: everything dataset-specific lives behind ``Dataset``,
everything metric-specific behind ``NoteMetric``. The one difference is where
the per-track step sits. Frame metrics receive pre-pooled arrays; note metrics
cannot pool their inputs (matching is per track), so this harness collects each
item's reference and estimate note lists -- the expensive part, running the
baseline and the segmenter -- then hands the whole nested list to each metric,
which loops tracks internally and returns one result.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
import sys

from stemify.baselines.base import PitchBaseline
from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.metrics import PRF, Fragmentation, MetricResult, NoteMetric
from stemify.evaluation.report import Report, jsonable
from stemify.notes import NoteEvent
from stemify.segmentation import HeuristicSegmenter, Segmenter
from stemify.transforms import AudioTransform, Identity

# PRF is the headline (what a tab is graded on); Fragmentation explains the
# false positives PRF only counts. Both read the same match, so they always
# agree on which estimates were wrong.
DEFAULT_METRICS: tuple[NoteMetric, ...] = (PRF(), Fragmentation())


@dataclass(frozen=True)
class NoteReport(Report):
    """Note-level result: each metric's output, plus per-item PRF for breakdown."""

    baseline: str
    segmenter: str
    dataset: str
    n_items: int
    overall: list[MetricResult]
    # How the audio was prepared before the baseline heard it. Defaulted so
    # existing callers keep working, but always recorded: a result is not
    # interpretable without knowing whether it came from a clean stem or a
    # separated one.
    transform: str = "identity"
    # (name, PRF scalars) per item -- for ranking the tracks the segmenter fails
    # outright, which a pooled F1 hides.
    per_item: list[tuple[str, dict[str, float]]] = field(default_factory=list)

    def scalar(self, metric: str, key: str) -> float | None:
        """One number by name, for comparing runs without parsing text."""
        for result in self.overall:
            if result.name == metric:
                return result.scalars.get(key)
        return None

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "segmenter": self.segmenter,
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
            f"\n=== {self.baseline} + {self.segmenter} on {self.dataset} "
            f"(note-level, {self.n_items:,} items{via}) ===",
        ]
        for result in self.overall:
            lines += [f"\n{result.name}", result.render()]

        # A pooled F1 hides a handful of tracks the segmenter fails outright,
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
    """Bind a dataset, a baseline and a segmenter; ``run()`` scores notes.

    ``segmenter`` turns the baseline's contour into notes. Pass ``None`` only for
    a baseline that emits notes *natively* (a transcription model), in which case
    ``predict_notes`` is used directly.
    """

    def __init__(
        self,
        dataset: Dataset,
        baseline: PitchBaseline,
        segmenter: Segmenter | None = None,
        metrics: Iterable[NoteMetric] = DEFAULT_METRICS,
        transform: AudioTransform | None = None,
        progress: bool = True,
    ) -> None:
        self.dataset = dataset
        self.baseline = baseline
        self.segmenter = segmenter if segmenter is not None else HeuristicSegmenter()
        self.metrics = list(metrics)
        # Preprocessing before the baseline hears the audio -- separation,
        # principally. The reference notes are ground truth and untouched by it;
        # only the estimate side sees the transformed audio, and both the pitch
        # contour and the onset detection read the same transformed path so they
        # stay consistent.
        self.transform = transform if transform is not None else Identity()
        # None means "use the baseline's native note output" -- set by passing
        # segmenter=..., unset means compose contour + segmenter.
        self._native = segmenter is None and not baseline.is_frame_level
        self.progress = progress

    def _estimate_notes(self, item: EvalItem) -> list[NoteEvent]:
        path = self.transform.apply(item.audio())
        if self._native:
            return self.baseline.predict_notes(path)
        return self.segmenter.segment(self.baseline.predict_f0(path), path)

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

        overall = [m.calculate(references, estimates) for m in self.metrics]

        # Per-item PRF for the worst-item breakdown: the same metric on a
        # one-track dataset. Reuse a PRF from the metrics list so its tolerances
        # match the reported numbers; fall back to a default if none is present.
        ranker = next((m for m in self.metrics if isinstance(m, PRF)), PRF())
        per_item = [
            (name, ranker.calculate([ref], [est]).scalars)
            for name, ref, est in zip(names, references, estimates)
        ]

        return NoteReport(
            baseline=self.baseline.name,
            segmenter="native" if self._native else self.segmenter.name,
            dataset=self.dataset.name,
            transform=self.transform.name,
            n_items=len(names),
            overall=overall,
            per_item=per_item,
        )