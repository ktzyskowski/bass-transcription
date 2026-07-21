"""``LearnedSegmenter``: a trained onset head in place of spectral flux.

This is the bridge from training back to evaluation. A ``FrameTagger`` with an
onset head produces a dense per-frame onset probability over the whole track (the
learned analogue of librosa's onset-strength envelope); peak-picking that curve
gives onset times, and those feed the *same* grouping the heuristic uses -- rests
and pitch changes still come from the pyin contour, only the same-pitch split cue
is now learned. So a ``LearnedSegmenter`` is a ``HeuristicSegmenter`` that
overrides one method, ``_onset_times``, and inherits everything else.

That also makes it exactly what the async validator scores: wrap a checkpoint,
run it through ``NoteEvaluation``, and its note-level F1 is the number the DSP
floor (73.1% stem / 64.7% demucs) has to be beaten by.

Onset-only for now: pitch and voicing still come from the pyin contour passed to
``segment``. When the model grows voicing/pitch heads, this is where their
predictions replace the contour.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from stemify.features import cqt, load_audio
from stemify.segmentation import HeuristicSegmenter
from stemify.training.dataset import frame_times_for
from stemify.training.model import FrameTagger
from stemify.training.trainer import best_device

ONSET_THRESHOLD = 0.5
MIN_ONSET_GAP_S = 0.03


def onset_times_from_probs(
    probs: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = ONSET_THRESHOLD,
    min_gap_s: float = MIN_ONSET_GAP_S,
) -> np.ndarray:
    """Peak-pick an onset-probability curve into onset times.

    A local maximum above ``threshold`` is an onset; ``min_gap_s`` enforces a
    refractory gap so one attack does not fire twice. Pure, so the peak logic is
    tested without a model or audio.
    """
    if probs.size == 0:
        return np.zeros(0)
    hop = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.01
    distance = max(1, round(min_gap_s / hop))
    peaks, _ = find_peaks(probs, height=threshold, distance=distance)
    return frame_times[peaks]


class LearnedSegmenter(HeuristicSegmenter):
    """Heuristic grouping with onsets from a trained onset head."""

    name = "learned"

    def __init__(
        self,
        model: FrameTagger,
        device: str | None = None,
        onset_threshold: float = ONSET_THRESHOLD,
        min_onset_gap_s: float = MIN_ONSET_GAP_S,
        **grouping: object,
    ) -> None:
        super().__init__(**grouping)  # min_note_s, pitch_split_semitones, merge_gap_s
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.onset_threshold = onset_threshold
        self.min_onset_gap_s = min_onset_gap_s

    @torch.no_grad()
    def onset_probs(self, audio_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """(per-frame onset probability, frame times) for a whole track.

        The model is fully convolutional over time, so the entire track goes
        through in one pass -- no windowing at inference, unlike the fixed-length
        crops training batches on.
        """
        features = cqt(load_audio(audio_path))
        x = torch.from_numpy(features).float()[None, None].to(self.device)  # (1,1,F,T)
        logits = self.model(x)["onset"]           # (1, T, 1)
        probs = torch.sigmoid(logits).squeeze(-1).squeeze(0).cpu().numpy()  # (T,)
        return probs, frame_times_for(features)

    def _onset_times(self, audio_path: str | Path) -> np.ndarray:
        probs, frame_times = self.onset_probs(audio_path)
        return onset_times_from_probs(
            probs, frame_times, self.onset_threshold, self.min_onset_gap_s
        )