"""``NeuralEncoder``: trained backbone + heads as the cue producer.

Runs a ``FrameTagger`` over a track's CQT to produce per-frame onset (and voicing,
when the model has that head) probabilities, pairs them with the pyin pitch contour,
and returns a ``Cues``. The model is fully convolutional over time, so the whole
track goes through in one pass -- no windowing at inference. Pitch is still pyin
until a learned pitch head replaces it (at which point the contour comes from a head
and this class stops calling the baseline -- the interface does not change).

The torch-bearing encoder, imported explicitly so the light DSP path never pulls it.
"""

import numpy as np
import torch

from stemify.baselines.pyin import PyinBaseline
from stemify.cues import Cues
from stemify.encoders.base import Encoder
from stemify.encoders.tagger import FrameTagger
from stemify.features import extract_features
from stemify.sources import Sources
from stemify.training.device import best_device


class NeuralEncoder(Encoder):
    """Neural cue producer: a ``FrameTagger``'s onset/voicing probs + pyin pitch."""

    name = "neural"

    def __init__(
        self,
        model: FrameTagger,
        device: str | None = None,
        baseline: PyinBaseline | None = None,
    ) -> None:
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.baseline = baseline or PyinBaseline()

    @torch.no_grad()
    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(onset probs, voicing probs) over a whole track's CQT. Voicing is ``None``
        when the model has no voicing head, so an onset-only checkpoint still
        encodes (the decoder then falls back to the contour's voicing)."""
        x = torch.from_numpy(np.ascontiguousarray(features)).float()
        out = self.model(x[None, None].to(self.device))  # (1, 1, F, T)
        onset = torch.sigmoid(out["onset"]).squeeze(-1).squeeze(0).cpu().numpy()
        voicing = None
        if "voicing" in out:
            voicing = torch.sigmoid(out["voicing"]).squeeze(-1).squeeze(0).cpu().numpy()
        return onset, voicing

    def encode(self, sources: Sources) -> Cues:
        feats = extract_features(sources.bass)
        onset, voicing = self._forward(feats.values)
        contour = self.baseline.predict_f0(sources.bass)
        return Cues(
            contour=contour,
            onset=onset,
            frame_times=feats.frame_times,
            voicing=voicing,
        )
