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
from stemify.parallel import parallel_map
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


def _warm_one(args: tuple[str, str]) -> None:
    """Pool worker: compute+cache one contour. Module-level and string-argued so it
    pickles under the spawn start method macOS uses."""
    audio_path, cache_dir = args
    cached_f0(PyinBaseline(), audio_path, Path(cache_dir))


def warm_contours(
    audio_paths: Sequence[str | Path], cache_dir: Path, workers: int
) -> None:
    """Precompute the pyin contour cache in parallel.

    pyin over tracks is embarrassingly parallel and dominates val setup (~10 s a
    track, single-threaded in ``OnsetValidator``'s construction loop), while 13 of
    14 cores sit idle. Because the on-disk cache is concurrency-safe (the atomic
    pid-``.partial`` write in ``cached_f0``), a process pool can fill it far faster;
    ``OnsetValidator`` then hits an all-cache and its loop is just CQT. Already-cached
    paths are skipped inside ``cached_f0``, so this resumes a partial cache cheaply."""
    items = [(str(p), str(cache_dir)) for p in audio_paths]
    parallel_map(_warm_one, items, workers=workers, chunksize=4)


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
                    contour=cached_f0(baseline, path, contour_cache),
                    notes=item.reference_notes(),
                )
            )

    def score(self, model: FrameTagger, device: str = "cpu") -> dict[str, float]:
        return score_examples(
            model, self.examples, self.grouper, self.threshold, self.min_gap_s, device
        )