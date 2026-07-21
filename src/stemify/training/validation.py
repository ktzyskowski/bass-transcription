"""Cached note-level validation for the async validator.

The async note-level eval is dominated by pyin, but pyin's pitch contour is fixed
across checkpoints -- only the onset model changes. ``OnsetValidator`` runs pyin
(and the CQT) over the val tracks *once* at construction and holds the results,
so each subsequent score is just a model forward, a peak-pick and the grouping --
seconds, not minutes. That is what makes async validation every few epochs cheap
enough to leave on.

``score_examples`` is the pure core (given precomputed examples), so the scoring
is tested with synthetic inputs and a fake model, no audio.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset
from stemify.evaluation.metrics import PRF, Fragmentation
from stemify.features import cqt, load_audio
from stemify.notes import NoteEvent
from stemify.segmentation import HeuristicSegmenter
from stemify.training.dataset import frame_times_for
from stemify.training.model import FrameTagger
from stemify.training.segmenter import (
    MIN_ONSET_GAP_S,
    ONSET_THRESHOLD,
    onset_times_from_probs,
)
from stemify.transforms import AudioTransform, Identity


@dataclass
class ValExample:
    """One val track, everything fixed across checkpoints precomputed."""

    features: torch.Tensor        # (1, 1, n_bins, T)
    frame_times: np.ndarray       # (T,)
    contour: Contour              # pyin pitch, fixed
    notes: list[NoteEvent]        # reference notes


@torch.no_grad()
def score_examples(
    model: FrameTagger,
    examples: Sequence[ValExample],
    grouper: HeuristicSegmenter,
    threshold: float,
    min_gap_s: float,
    device: str = "cpu",
) -> dict[str, float]:
    """Score a model over precomputed val examples: model onsets -> grouping ->
    note-level P/R/F and fragmentation. Pure given ``examples``."""
    model = model.to(device).eval()
    refs: list[list[NoteEvent]] = []
    ests: list[list[NoteEvent]] = []
    for ex in examples:
        logits = model(ex.features.to(device))["onset"]           # (1, T, 1)
        probs = torch.sigmoid(logits).squeeze(-1).squeeze(0).cpu().numpy()
        onsets = onset_times_from_probs(probs, ex.frame_times, threshold, min_gap_s)
        ests.append(grouper.group(ex.contour, onsets))
        refs.append(ex.notes)
    prf = PRF().calculate(refs, ests).scalars
    frag = Fragmentation().calculate(refs, ests).scalars
    return {
        "note_f1": prf["f1"],
        "note_precision": prf["precision"],
        "note_recall": prf["recall"],
        "fragmentation": float(frag["fragmentation"]),
    }


class OnsetValidator:
    """Precompute pyin contours + CQT + reference notes once; score models cheaply."""

    def __init__(
        self,
        dataset: Dataset,
        transform: AudioTransform | None = None,
        threshold: float = ONSET_THRESHOLD,
        min_gap_s: float = MIN_ONSET_GAP_S,
        **grouping: object,
    ) -> None:
        self.threshold = threshold
        self.min_gap_s = min_gap_s
        self.grouper = HeuristicSegmenter(**grouping)
        transform = transform or Identity()
        baseline = PyinBaseline()
        self.examples: list[ValExample] = []
        for item in dataset:
            path = transform.apply(item.audio())
            features = cqt(load_audio(path))
            self.examples.append(
                ValExample(
                    features=torch.from_numpy(features).float()[None, None],
                    frame_times=frame_times_for(features),
                    contour=baseline.predict_f0(path),
                    notes=item.reference_notes(),
                )
            )

    def score(self, model: FrameTagger, device: str = "cpu") -> dict[str, float]:
        return score_examples(
            model, self.examples, self.grouper, self.threshold, self.min_gap_s, device
        )