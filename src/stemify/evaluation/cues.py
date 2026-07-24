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

The pooling/scoring is ``metrics.cues.score_cues`` (a leaf the async validator shares,
so the in-training numbers cannot drift from this harness). This module only turns a
dataset item + the encoder's ``Cues`` into the aligned per-channel pairs it wants:
onset target from the reference notes, voicing target from the reference contour, and
the contour resampled onto the reference grid for pitch.
"""

import sys
from dataclasses import dataclass

import numpy as np

from stemify.cues import Cues
from stemify.data.base import Dataset, EvalItem
from stemify.encoders.base import Encoder
from stemify.evaluation.alignment import align
from stemify.evaluation.report import Report, jsonable
from stemify.metrics import CuesChannels, MetricResult, score_cues
from stemify.sources import Sources
from stemify.training.labels import onset_targets, voicing_targets
from stemify.transforms import AudioTransform, Identity


def channels_for(cues: Cues, item: EvalItem) -> CuesChannels:
    """The aligned per-channel pairs for one track: what ``score_cues`` scores.

    Onset target is impulses on the frame grid at the reference onsets; voicing target
    is the reference contour's voiced mask resampled onto the frame grid; pitch pairs
    the encoder's own contour (resampled onto the reference grid, voicing-aware) with
    the reference. Voicing is dropped when the encoder has no voicing head."""
    reference = item.reference()
    onset_true = onset_targets(
        np.array([n.onset for n in item.reference_notes()], dtype=float),
        cues.frame_times,
    )
    voicing = None
    if cues.voicing is not None:
        voi_true = voicing_targets(reference.times, reference.voiced, cues.frame_times)
        voicing = (cues.voicing, voi_true)
    ref_hz, est_hz = align(reference, cues.contour)
    return CuesChannels(
        onset=(cues.onset, onset_true),
        voicing=voicing,
        pitch=(est_hz, ref_hz),
    )


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

    def _channels(self, item: EvalItem) -> CuesChannels:
        path = self.transform.apply(item.audio())
        cues = self.encoder.encode(Sources(bass=path))
        return channels_for(cues, item)

    def run(self) -> CuesReport:
        items = list(self.dataset)
        tracks: list[CuesChannels] = []
        for i, item in enumerate(items, 1):
            tracks.append(self._channels(item))
            if self.progress:
                print(f"  {i}/{len(items)} {item.name[:44]:<44}", end="\r", file=sys.stderr)
        if self.progress:
            print(file=sys.stderr)

        if not tracks:
            raise ValueError(
                f"No items to evaluate for {self.dataset.name}. Check the dataset "
                "root, and whether filters (polyphony, split, limit) excluded "
                "everything."
            )

        return CuesReport(
            encoder=self.encoder.name,
            dataset=self.dataset.name,
            transform=self.transform.name,
            n_items=len(tracks),
            overall=score_cues(tracks),
        )
