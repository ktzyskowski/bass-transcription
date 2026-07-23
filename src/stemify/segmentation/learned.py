"""``LearnedSegmenter``: trained heads in place of DSP cues.

The bridge from training back to evaluation. A ``FrameTagger`` produces dense
per-frame probabilities over the whole track -- an onset curve (the learned
analogue of librosa's onset-strength envelope) and, when the model has the head, a
voicing curve. Peak-picking the onset curve gives onset times; thresholding the
voicing curve gives a note/rest mask; both feed the *same* grouping the heuristic
uses. Pitch still comes from the pyin contour. So a ``LearnedSegmenter`` is a
``HeuristicSegmenter`` that swaps the DSP onset (and, if present, voicing) cues for
learned ones and inherits the grouping unchanged.

The whole learned->notes path lives in one place, ``segment_features``: the async
validator hands it precomputed CQT (no audio->CQT recompute), deployment hands it
a fresh CQT via ``segment`` -- one code path, so eval and deploy cannot drift, and
callers only ever see ``NoteEvent``s.

An onset-only checkpoint still works: with no voicing head, the voicing cue falls
back to the pyin contour's own ``freqs > 0``, exactly the previous behaviour.
"""

from pathlib import Path

import numpy as np
import torch

from stemify.contour import Contour
from stemify.decoders.heuristic import (
    MIN_ONSET_GAP_S,
    ONSET_THRESHOLD,
    VOICING_THRESHOLD,
    onset_times_from_probs,
)
from stemify.encoders.tagger import FrameTagger
from stemify.features import Features, extract_features
from stemify.notes import NoteEvent
from stemify.segmentation.heuristic import HeuristicSegmenter
from stemify.training.device import best_device
from stemify.training.labels import resample_mask


class LearnedSegmenter(HeuristicSegmenter):
    """Heuristic grouping with onsets (and optionally voicing) from a trained head."""

    name = "learned"

    def __init__(
        self,
        model: FrameTagger,
        device: str | None = None,
        onset_threshold: float = ONSET_THRESHOLD,
        min_onset_gap_s: float = MIN_ONSET_GAP_S,
        voicing_threshold: float = VOICING_THRESHOLD,
        **grouping: object,
    ) -> None:
        super().__init__(**grouping)  # min_note_s, pitch_split_semitones, merge_gap_s
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.onset_threshold = onset_threshold
        self.min_onset_gap_s = min_onset_gap_s
        self.voicing_threshold = voicing_threshold

    @torch.no_grad()
    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(onset probs, voicing probs) over a whole track from CQT features.

        The model is fully convolutional over time, so the entire track goes through
        in one pass -- no windowing at inference, unlike the crops training batches
        on. Voicing is ``None`` when the model has no voicing head, so an onset-only
        checkpoint still segments (grouping falls back to the contour's voicing).
        """
        x = torch.from_numpy(np.ascontiguousarray(features)).float()
        out = self.model(x[None, None].to(self.device))  # (1, 1, F, T)
        onset = torch.sigmoid(out["onset"]).squeeze(-1).squeeze(0).cpu().numpy()
        voicing = None
        if "voicing" in out:
            voicing = torch.sigmoid(out["voicing"]).squeeze(-1).squeeze(0).cpu().numpy()
        return onset, voicing

    def onset_probs(self, audio_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """(per-frame onset probability, frame times) for a whole track."""
        feats = extract_features(audio_path)
        onset, _ = self._forward(feats.values)
        return onset, feats.frame_times

    def voicing_probs(self, features: Features) -> np.ndarray | None:
        """Raw per-frame P(voiced) over a whole track, or ``None`` if the model has
        no voicing head. The un-thresholded signal a threshold-free metric (voicing
        AUROC) needs, distinct from the thresholded mask ``segment_features`` uses."""
        _, voicing = self._forward(features.values)
        return voicing

    def _onset_times(self, audio_path: str | Path) -> np.ndarray:
        probs, frame_times = self.onset_probs(audio_path)
        return onset_times_from_probs(
            probs, frame_times, self.onset_threshold, self.min_onset_gap_s
        )

    def segment_features(self, contour: Contour, features: Features) -> list[NoteEvent]:
        """Notes from precomputed CQT features -- the cache-friendly entry the
        validator uses. Owns the whole learned->notes path: peak-pick onsets,
        threshold voicing (if the model has that head) and align it to the contour
        grid, then group. ``segment`` is this over a freshly computed CQT."""
        onset, voicing = self._forward(features.values)
        onsets = onset_times_from_probs(
            onset, features.frame_times, self.onset_threshold, self.min_onset_gap_s
        )
        voiced = None
        if voicing is not None:
            mask = voicing > self.voicing_threshold
            voiced = resample_mask(features.frame_times, mask, contour.times).astype(bool)
        return self.group(contour, onsets, voiced=voiced)

    def segment(self, contour: Contour, audio_path: str | Path) -> list[NoteEvent]:
        return self.segment_features(contour, extract_features(audio_path))
