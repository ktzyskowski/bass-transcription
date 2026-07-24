"""Cues-level evaluation: score an ``Encoder``'s output, decoder-independent.

    CuesEvaluation(SlakhDataset("data/raw/slakh2100"), DSPEncoder()).run()

``NoteEvaluation`` scores a whole ``Segmenter`` -- "did you find the right notes".
This scores the *encoder* alone: given the per-frame ``Cues`` it produces, how good
is the evidence a decoder has to work with? It is the altitude below the notes, and
it maps onto the three channels of a ``Cues``:

    onset    OnsetPR         is the onset curve sharp and un-ringing (over-seg, early)
    voicing  VoicingAUROC    does the voicing head rank voiced above rest
    pitch    PitchAccuracy   is the contour right (RPA/RCA) -- checkpoint-invariant
                             while pitch stays pyin

Run alongside ``NoteEvaluation`` it localises a regression: if note-F1 drops but the
Cues metrics hold, the *decoder* regressed (a duration-potential bug, a bad merge);
if both drop, the *encoder* did. That attribution is exactly what a pooled note-F1
cannot give, and it is the principled home for frame-level pitch scoring now that the
pitch-tracker harness is gone -- measured on the real encoder output, not a baseline.

The scoring core (``score_cues``) is pure over precomputed ``ScoredCues`` -- the same
split ``NoteEvaluation``/``score_examples`` use -- so the async validator scores the
encoder every epoch from its cached features through this exact code, no drift.

Frame metrics pool by concatenation (the ``FrameMetric`` contract), so each channel's
frames are gathered across every track and scored once, frame-weighted by construction.
A channel is scored only when every track supplies its truth: onset always (from the
reference notes), voicing when the encoder has a voicing head and a per-frame voiced
label is supplied, pitch when a reference contour is supplied (the offline path; the
async validator omits it, since pyin pitch does not move between checkpoints).
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.data.base import Dataset, EvalItem
from stemify.encoders.base import Encoder
from stemify.evaluation.alignment import align
from stemify.evaluation.report import Report, jsonable
from stemify.metrics import MetricResult, PitchAccuracy
from stemify.metrics.frame import OnsetPR, VoicingAUROC
from stemify.notes import NoteEvent
from stemify.sources import Sources
from stemify.training.labels import onset_targets, voicing_targets
from stemify.transforms import AudioTransform, Identity


@dataclass(frozen=True)
class ScoredCues:
    """One track's encoder output plus the reference truth to score it against.

    ``cues`` is the encoder's whole-track output. ``reference_notes`` gives the onset
    channel its target (impulses on the frame grid). ``voicing_target`` is a per-frame
    voiced label already on ``cues.frame_times`` (the offline path builds it from the
    reference contour; the async validator passes its cached training target) -- kept
    explicit because its truth source legitimately differs between the two callers.
    ``reference_contour`` drives the pitch channel and is optional: absent (async),
    pitch is simply not scored."""

    cues: Cues
    reference_notes: list[NoteEvent]
    voicing_target: np.ndarray | None = None
    reference_contour: Contour | None = None


def score_cues(items: Sequence[ScoredCues]) -> list[MetricResult]:
    """Score whichever channels every track can supply, pooled across tracks.

    Pure given ``items`` -- no dataset, no encoder, no audio -- so both the offline
    ``CuesEvaluation`` and the async validator score through it identically."""
    results: list[MetricResult] = []
    if not items:
        return results

    # Onset -- always available (reference notes give the onsets).
    onset_true = np.concatenate(
        [
            onset_targets(
                np.array([n.onset for n in it.reference_notes], dtype=float),
                it.cues.frame_times,
            )
            for it in items
        ]
    )
    onset_pred = np.concatenate([it.cues.onset for it in items])
    results.append(OnsetPR().calculate(onset_true, onset_pred))

    # Voicing -- only if the encoder has the head and every track carries a label.
    if all(it.cues.voicing is not None and it.voicing_target is not None for it in items):
        voi_true = np.concatenate([it.voicing_target for it in items])
        voi_pred = np.concatenate([it.cues.voicing for it in items])
        results.append(VoicingAUROC().calculate(voi_true, voi_pred))

    # Pitch -- only when a reference contour is supplied (the offline path). The
    # estimate is the encoder's own contour, resampled onto the reference grid.
    if all(it.reference_contour is not None for it in items):
        parts = [align(it.reference_contour, it.cues.contour) for it in items]
        pitch_true = np.concatenate([t for t, _ in parts])
        pitch_pred = np.concatenate([p for _, p in parts])
        results.append(PitchAccuracy().calculate(pitch_true, pitch_pred))

    return results


@dataclass(frozen=True)
class CuesReport(Report):
    """Encoder-level result: each scored channel's metric output."""

    encoder: str
    dataset: str
    n_items: int
    overall: list[MetricResult]
    transform: str = "identity"

    def scalar(self, metric: str, key: str) -> float | None:
        """One number by name, for comparing runs without parsing text."""
        for result in self.overall:
            if result.name == metric:
                return result.scalars.get(key)
        return None

    def to_dict(self) -> dict:
        return {
            "encoder": self.encoder,
            "dataset": self.dataset,
            "transform": self.transform,
            "level": "cues",
            "n_items": self.n_items,
            "overall": {
                r.name: {
                    "scalars": {k: jsonable(v) for k, v in r.scalars.items()},
                    "arrays": {k: jsonable(v) for k, v in r.arrays.items()},
                }
                for r in self.overall
            },
        }

    def render(self, worst: int = 10) -> str:
        via = "" if self.transform == "identity" else f", audio via {self.transform}"
        lines = [
            (
                f"\n=== {self.encoder} on {self.dataset} "
                f"(cues-level, {self.n_items:,} items{via}) ==="
            ),
        ]
        for result in self.overall:
            lines += [f"\n{result.name}", result.render()]
        return "\n".join(lines)


class CuesEvaluation:
    """Bind a dataset and an ``Encoder``; ``run()`` scores the cues it produces.

    ``transform`` preprocesses the audio before the encoder hears it (separation,
    principally); the reference is ground truth and untouched by it.
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: Encoder,
        transform: AudioTransform | None = None,
        progress: bool = True,
    ) -> None:
        self.dataset = dataset
        self.encoder = encoder
        self.transform = transform if transform is not None else Identity()
        self.progress = progress

    def _scored(self, item: EvalItem) -> ScoredCues:
        path = self.transform.apply(item.audio())
        cues = self.encoder.encode(Sources(bass=path))
        reference = item.reference()
        return ScoredCues(
            cues=cues,
            reference_notes=item.reference_notes(),
            voicing_target=voicing_targets(
                reference.times, reference.voiced, cues.frame_times
            ),
            reference_contour=reference,
        )

    def run(self) -> CuesReport:
        items = list(self.dataset)
        scored: list[ScoredCues] = []
        for i, item in enumerate(items, 1):
            scored.append(self._scored(item))
            if self.progress:
                print(f"  {i}/{len(items)} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)

        if not scored:
            raise ValueError(
                f"No items to evaluate for {self.dataset.name}. Check the dataset "
                "root, and whether filters (polyphony, split, limit) excluded "
                "everything."
            )

        return CuesReport(
            encoder=self.encoder.name,
            dataset=self.dataset.name,
            transform=self.transform.name,
            n_items=len(scored),
            overall=score_cues(scored),
        )
