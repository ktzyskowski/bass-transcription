"""``NeuralEncoder``: trained backbone + heads as the cue producer.

Runs a ``FrameTagger`` over a track's CQT to produce per-frame onset (and voicing,
when the model has that head) probabilities, pairs them with the pyin pitch contour,
and returns a ``Cues``. The model is fully convolutional over time, so the whole
track goes through in one pass -- no windowing at inference. Pitch comes from an
injected ``PitchEstimator`` (``PyinPitch`` by default); a learned pitch head is just
another ``PitchEstimator`` that drops in here with no change to this class.

The torch-bearing encoder, imported explicitly so the light DSP path never pulls it.
"""

import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.encoders.base import Encoder
from stemify.encoders.tagger import FrameTagger
from stemify.features import FrameFeatures, extract_features
from stemify.pitch import PitchEstimator, PyinPitch
from stemify.sources import Sources
from stemify.training.device import best_device


class NeuralEncoder(Encoder):
    """Neural cue producer: a ``FrameTagger``'s onset/voicing probs + pyin pitch."""

    name = "neural"

    def __init__(
        self,
        model: FrameTagger,
        device: str | None = None,
        pitch: PitchEstimator | None = None,
    ) -> None:
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.pitch = pitch or PyinPitch()

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
        contour = self.pitch.estimate(sources.bass)
        return self.encode_features(contour, feats)

    def encode_features(self, contour: Contour, features: FrameFeatures) -> Cues:
        """``Cues`` from a precomputed CQT and contour -- the cache-friendly path the
        async validator uses (no CQT/pyin recompute, both already cached per track).
        ``encode`` is this over a freshly computed CQT and pyin contour."""
        onset, voicing = self._forward(features.values)
        return Cues(
            contour=contour,
            onset=onset,
            frame_times=features.frame_times,
            voicing=voicing,
        )

    def voicing_probs(self, features: FrameFeatures) -> np.ndarray | None:
        """Raw per-frame P(voiced) over a track, or ``None`` with no voicing head --
        the un-thresholded signal the threshold-free voicing AUROC metric needs,
        distinct from the thresholded mask the decoder places rests with."""
        _, voicing = self._forward(features.values)
        return voicing
