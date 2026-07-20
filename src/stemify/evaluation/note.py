"""Note-level evaluation: precision / recall / F1 over note events.

The frame-level harness (``Evaluation``) asks "what pitch at each instant". This
asks "did you find the right notes" -- the metric a tab is actually graded on,
and the one a bass-transcription collaborator's F1 numbers use, so it is what
makes those numbers comparable.

The data flow is genuinely different from frame-level, which is why this is a
separate harness rather than another ``Metric``:

    reference notes   (dataset, note-native)
    estimated notes   (segmenter over a baseline's contour, or a native model)
    match             (mir_eval.transcription, onset + pitch tolerance)
    pool COUNTS        tp/fp/fn summed across items, then P/R/F once

Pooling counts rather than averaging per-item F1 is the note-level analogue of
frame pooling: a track with 200 notes weighs 10x one with 20, and a track's F1
is undefined when it has no notes -- summing tp/fp/fn sidesteps both.

Matching ignores offsets (``offset_ratio=None``): a bass note's end is fuzzy
(decay, release), its start is not, so onset + pitch is the honest criterion.
Onset tolerance defaults to 50 ms and pitch tolerance to 50 cents, the
``mir_eval`` conventions, matching the frame-level protocol.
"""

import sys
from dataclasses import dataclass, field

import mir_eval
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.data.base import Dataset, EvalItem
from stemify.evaluation.report import Report
from stemify.notes import NoteEvent
from stemify.segmentation import HeuristicSegmenter, Segmenter

ONSET_TOLERANCE_S = 0.05
PITCH_TOLERANCE_CENTS = 50.0


def _to_arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    """(intervals Nx2 seconds, pitches N in Hz), the mir_eval note format."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.onset, n.offset] for n in notes])
    pitches = _midi_to_hz(np.array([n.midi_pitch for n in notes]))
    return intervals, pitches


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def count_matches(
    ref: list[NoteEvent],
    est: list[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE_S,
    pitch_tolerance: float = PITCH_TOLERANCE_CENTS,
) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one item.

    An estimated note is a true positive if some reference note matches it in
    onset (within tolerance) and pitch; each reference note matches at most one
    estimate. Empty sides are handled directly -- ``mir_eval.match_notes``
    expects non-empty inputs.
    """
    if not ref and not est:
        return 0, 0, 0
    if not ref:
        return 0, len(est), 0
    if not est:
        return 0, 0, len(ref)

    ref_int, ref_hz = _to_arrays(ref)
    est_int, est_hz = _to_arrays(est)
    matches = mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,  # onset + pitch only; bass offsets are fuzzy
    )
    tp = len(matches)
    return tp, len(est) - tp, len(ref) - tp


@dataclass(frozen=True)
class NoteReport(Report):
    """What one baseline+segmenter scored on one dataset, note-level."""

    baseline: str
    segmenter: str
    dataset: str
    tp: int
    fp: int
    fn: int
    per_item: list[tuple[str, int, int, int]] = field(default_factory=list)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    @property
    def precision(self) -> float:
        return self._prf(self.tp, self.fp, self.fn)[0]

    @property
    def recall(self) -> float:
        return self._prf(self.tp, self.fp, self.fn)[1]

    @property
    def f1(self) -> float:
        return self._prf(self.tp, self.fp, self.fn)[2]

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "segmenter": self.segmenter,
            "dataset": self.dataset,
            "level": "note",
            "onset_tolerance_s": ONSET_TOLERANCE_S,
            "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "per_item": [
                {"name": n, "tp": tp, "fp": fp, "fn": fn}
                for n, tp, fp, fn in self.per_item
            ],
        }

    def render(self, worst: int = 10) -> str:
        p, r, f = self.precision, self.recall, self.f1
        lines = [
            f"\n=== {self.baseline} + {self.segmenter} on {self.dataset} "
            f"(note-level, {len(self.per_item)} items) ===",
            f"  reference / estimated notes  {self.tp + self.fn:,} / {self.tp + self.fp:,}",
            f"  precision                    {p:>7.1%}",
            f"  recall                       {r:>7.1%}",
            f"  F1                           {f:>7.1%}",
            f"  tp / fp / fn                 {self.tp:,} / {self.fp:,} / {self.fn:,}",
        ]
        # Worst items by F1 -- surfaces tracks the segmenter fails outright,
        # which a pooled F1 hides.
        if len(self.per_item) > 1 and worst:
            ranked = sorted(
                self.per_item, key=lambda x: self._prf(x[1], x[2], x[3])[2]
            )[:worst]
            lines.append(f"\n  worst {len(ranked)} items by F1")
            lines.append(f"  {'F1':>7} {'tp':>5} {'fp':>5} {'fn':>5}  item")
            for name, tp, fp, fn in ranked:
                lines.append(
                    f"  {self._prf(tp, fp, fn)[2]:>7.1%} {tp:>5} {fp:>5} {fn:>5}  {name[:44]}"
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
        onset_tolerance: float = ONSET_TOLERANCE_S,
        pitch_tolerance: float = PITCH_TOLERANCE_CENTS,
        progress: bool = True,
    ) -> None:
        self.dataset = dataset
        self.baseline = baseline
        self.segmenter = segmenter if segmenter is not None else HeuristicSegmenter()
        # None means "use the baseline's native note output" -- set by passing
        # segmenter=..., unset means compose contour + segmenter.
        self._native = segmenter is None and not baseline.is_frame_level
        self.onset_tolerance = onset_tolerance
        self.pitch_tolerance = pitch_tolerance
        self.progress = progress

    def _estimate_notes(self, item: EvalItem) -> list[NoteEvent]:
        path = item.audio()
        if self._native:
            return self.baseline.predict_notes(path)
        return self.segmenter.segment(self.baseline.predict_f0(path), path)

    def run(self) -> NoteReport:
        items = list(self.dataset)
        per_item: list[tuple[str, int, int, int]] = []
        for i, item in enumerate(items, 1):
            ref = item.reference_notes()
            est = self._estimate_notes(item)
            tp, fp, fn = count_matches(
                ref, est, self.onset_tolerance, self.pitch_tolerance
            )
            per_item.append((item.name, tp, fp, fn))
            if self.progress:
                print(f"  {i}/{len(items)} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)

        if not per_item:
            raise ValueError(f"No items to evaluate for {self.dataset.name}.")

        return NoteReport(
            baseline=self.baseline.name,
            segmenter="native" if self._native else self.segmenter.name,
            dataset=self.dataset.name,
            tp=sum(x[1] for x in per_item),
            fp=sum(x[2] for x in per_item),
            fn=sum(x[3] for x in per_item),
            per_item=per_item,
        )