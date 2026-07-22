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

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stemify.baselines.pyin import PyinBaseline
from stemify.contour import Contour
from stemify.data.base import Dataset
from stemify.evaluation.metrics import PRF, Fragmentation
from stemify.features import cqt, load_audio
from stemify.notes import NoteEvent
from stemify.segmentation import HeuristicSegmenter
from stemify.training.dataset import (
    TrackExample,
    example_path,
    frame_times_for,
    prepare_examples,
)
from stemify.training.model import FrameTagger
from stemify.training.segmenter import (
    MIN_ONSET_GAP_S,
    ONSET_THRESHOLD,
    onset_times_from_probs,
)
from stemify.transforms import AudioTransform, Identity


def cached_f0(
    baseline: PyinBaseline, audio_path: str | Path, cache_dir: Path | None
) -> Contour:
    """pyin contour for an audio file, memoised to disk.

    The contour is a pure function of the (already transform-resolved) audio and
    pyin's fixed params, so it is computed once per file and reused across every
    checkpoint *and every future run* -- the same disk-cache rationale as the CQT
    features, and the reason a large val set becomes affordable. Keyed on the
    resolved audio path, so each condition's file (clean stem, mix, separated bass)
    caches independently and never collides. ``cache_dir=None`` disables caching.
    """
    if cache_dir is None:
        return baseline.predict_f0(audio_path)
    audio_path = Path(audio_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{baseline.name}\0{audio_path.resolve()}".encode()).hexdigest()
    dst = cache_dir / f"{audio_path.stem}-{key[:16]}.npz"
    if dst.exists():
        data = np.load(dst)
        conf = data["confidence"] if "confidence" in data.files else None
        return Contour(times=data["times"], freqs=data["freqs"], confidence=conf)

    contour = baseline.predict_f0(audio_path)
    arrays = {"times": contour.times, "freqs": contour.freqs}
    if contour.confidence is not None:
        arrays["confidence"] = contour.confidence
    # Write-aside-and-rename: an interrupted pyin must not leave a truncated npz
    # that later reads back as a cache hit (same guard as the transforms cache).
    partial = dst.with_name(f"{dst.stem}.{os.getpid()}.partial.npz")
    try:
        np.savez(partial, **arrays)
        partial.replace(dst)
    finally:
        partial.unlink(missing_ok=True)
    return contour


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
        contour_cache: Path | None = None,
        feature_cache: Path | None = None,
        threshold: float = ONSET_THRESHOLD,
        min_gap_s: float = MIN_ONSET_GAP_S,
        **grouping: object,
    ) -> None:
        self.threshold = threshold
        self.min_gap_s = min_gap_s
        self.grouper = HeuristicSegmenter(**grouping)
        transform = transform or Identity()
        baseline = PyinBaseline()

        # The CQT is the same the training pipeline already caches. With
        # ``feature_cache`` set, populate it once (``prepare_examples``, disk-cached
        # per (track, transform)) and read features back rather than recomputing in
        # memory every launch -- and reuse the stem cache the BCE loader already
        # built instead of recomputing it a second time.
        if feature_cache is not None:
            prepare_examples(dataset, ["onset"], feature_cache, transform)

        self.examples: list[ValExample] = []
        for item in dataset:
            path = transform.apply(item.audio())
            if feature_cache is not None:
                cached = TrackExample.load(
                    example_path(feature_cache, item.name, transform.name)
                )
                features, frame_times = cached.features, cached.frame_times
            else:
                features = cqt(load_audio(path))
                frame_times = frame_times_for(features)
            self.examples.append(
                ValExample(
                    features=torch.from_numpy(features).float()[None, None],
                    frame_times=frame_times,
                    contour=cached_f0(baseline, path, contour_cache),
                    notes=item.reference_notes(),
                )
            )

    def score(self, model: FrameTagger, device: str = "cpu") -> dict[str, float]:
        return score_examples(
            model, self.examples, self.grouper, self.threshold, self.min_gap_s, device
        )