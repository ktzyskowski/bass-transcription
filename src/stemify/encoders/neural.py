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
from stemify.models.tagger import FrameTagger
from stemify.features import FrameFeatures, extract_features, stack_channels
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

    @property
    def in_channels(self) -> int:
        """How many stacked renderings the wrapped model expects.

        Defaulted rather than required: a single CQT was the only input a model
        could have before mix-conditioning existed, so a backbone that does not
        declare a width is one -- which keeps every earlier checkpoint, and the
        duck-typed models the tests drive, working unchanged.
        """
        return getattr(self.model, "in_channels", 1)

    @torch.no_grad()
    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(onset probs, voicing probs) over a whole track's CQT. Voicing is ``None``
        when the model has no voicing head, so an onset-only checkpoint still
        encodes (the decoder then falls back to the contour's voicing).

        ``features`` is ``(n_bins, T)`` or, for a mix-conditioned model,
        ``(C, n_bins, T)``; the bare CQT gains the channel axis here so that both
        shapes reach the conv front-end as ``(1, C, F, T)``."""
        x = torch.from_numpy(np.ascontiguousarray(features)).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)                           # (F, T) -> (1, F, T)
        if x.shape[0] != self.in_channels:
            raise ValueError(
                f"model expects {self.in_channels} input channel(s), got "
                f"{x.shape[0]}. A mix-conditioned checkpoint must be encoded over "
                "the same channels it was trained on."
            )
        out = self.model(x[None].to(self.device))        # (1, C, F, T)
        onset = torch.sigmoid(out["onset"]).squeeze(-1).squeeze(0).cpu().numpy()
        voicing = None
        if "voicing" in out:
            voicing = torch.sigmoid(out["voicing"]).squeeze(-1).squeeze(0).cpu().numpy()
        return onset, voicing

    def encode(self, sources: Sources) -> Cues:
        """``Cues`` for a track, computing whatever the model's input needs.

        A mix-conditioned checkpoint reads a second channel from ``sources.mix``.
        No transform is applied to either: at deployment ``bass`` is already
        whatever the caller separated, so the transforms that build training
        channels have no role here -- they exist to turn a corpus's mixture into a
        bass, which is a step that has already happened by the time a ``Sources``
        arrives. Pitch and the targets always describe channel 0.
        """
        channels = [extract_features(sources.bass)]
        if self.in_channels == 2:
            channels.append(extract_features(sources.role("mix")))
        elif self.in_channels != 1:
            raise ValueError(
                f"no deployment channel layout for a {self.in_channels}-channel "
                "model; only (bass,) and (bass, mix) are defined"
            )
        contour = self.pitch.estimate(sources.bass)
        return self.encode_features(contour, stack_channels(channels))

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
