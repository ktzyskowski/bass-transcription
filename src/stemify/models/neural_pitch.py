"""``NeuralPitch``: a trained ``FrequencyHead`` as a standalone pitch source.

The learned peer of ``PyinPitch``. It satisfies the same ``estimate(audio_path)``
contract, so anywhere pyin is used it can be substituted -- including inside
``ClassicalEncoder``, which pairs spectral-flux onsets with whatever pitch source it is
given, and in the offline harness that scores pitch quality on its own.

**It is not what ``LearnedEncoder`` uses.** That encoder has already computed the CQT
and run the backbone by the time it wants a contour, so it reads the pitch logits out
of the forward pass it just did rather than paying for both again here. This class
exists for the callers that have a *path* and nothing else. Both routes end in
``models.heads.decode_contour``, so there is one decode with two entry points, not two
implementations that can disagree -- which is the property that makes substituting one
for the other meaningful.

It lives here rather than beside ``PyinPitch`` in ``core.pitch`` because it holds a
``CueNet``: the contract is core's, the implementation is this layer's. That is what
lets the DSP path take a ``PitchEstimator`` without torch coming with it.
"""

from pathlib import Path

import numpy as np
import torch

from stemify.core.contour import Contour
from stemify.platform.device import best_device
from stemify.audio.features import extract_features
from stemify.models.heads import CONTOUR_VOICING_THRESHOLD, decode_contour
from stemify.models.cuenet import CueNet
from stemify.core.pitch.base import PitchEstimator

class NeuralPitch(PitchEstimator):
    """A ``CueNet``'s pitch head as an audio-in pitch source."""

    name = "neural"

    def __init__(
        self,
        model: CueNet,
        device: str | None = None,
        voicing_threshold: float = CONTOUR_VOICING_THRESHOLD,
    ) -> None:
        if not getattr(model, "frequency_heads", None) or "pitch" not in model.frequency_heads:
            raise ValueError(
                "this model has no pitch head, so it cannot be a pitch source. Load a "
                "checkpoint trained with --pitch, or use PyinPitch."
            )
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.voicing_threshold = voicing_threshold

    @torch.no_grad()
    def estimate(self, audio_path: str | Path) -> Contour:
        """Contour for a file: CQT, one forward pass, decode.

        Single-channel only. A mix-conditioned model needs a second rendering that a
        lone path cannot supply, and guessing one (duplicating the bass into the mix
        channel) would quietly produce a contour from an input the model never saw in
        training. Those callers should go through ``LearnedEncoder``, which is handed a
        ``Sources`` and can resolve every channel.
        """
        width = getattr(self.model, "in_channels", 1)
        if width != 1:
            raise ValueError(
                f"NeuralPitch needs a 1-channel model, got {width}. A path names one "
                "rendering; use LearnedEncoder for a mix-conditioned checkpoint."
            )
        features = extract_features(audio_path)
        x = torch.from_numpy(np.ascontiguousarray(features.values)).float()
        out = self.model(x[None, None].to(self.device))    # (1, 1, F, T)

        logits = out["pitch"].squeeze(0).cpu().numpy()     # (T, n_bins)
        voiced = None
        if "voicing" in out:
            voicing = torch.sigmoid(out["voicing"]).squeeze(-1).squeeze(0).cpu().numpy()
            voiced = voicing > self.voicing_threshold
        return decode_contour(logits, features.frame_times, voiced)
