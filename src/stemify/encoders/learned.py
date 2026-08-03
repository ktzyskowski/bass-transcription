"""``LearnedEncoder``: trained backbone + heads as the cue producer.

Runs a ``CueNet`` over a track's CQT to produce per-frame onset (and voicing,
when the model has that head) probabilities, pairs them with a pitch contour, and
returns a ``Cues``. The model is fully convolutional over time, so the whole track
goes through in one pass -- no windowing at inference.

Pitch comes from one of two places. A model carrying a ``FrequencyHead`` produces it
in the *same forward pass* as onset and voicing; otherwise it falls back to the
injected ``PitchEstimator`` (``PyinPitch`` by default).

The learned head *does* also exist as a ``PitchEstimator`` -- ``pitch.neural.NeuralPitch``
-- for callers holding only a path. This class does not use it, for one reason: by the
time it wants a contour it has already computed the CQT and run the backbone, so going
back out through ``estimate(audio_path)`` would pay for both again. It reads the pitch
logits out of the forward pass it just did instead.

That is an optimisation, not a second implementation. Both routes end in
``models.heads.decode_contour``, so "the learned contour" means the same thing however
it was reached -- which is what makes the two substitutable rather than merely similar.

``use_pyin=True`` forces the DSP contour even on a model that has a head, which is how
an A/B between the two is run.

The torch-bearing encoder, imported explicitly so the light DSP path never pulls it.
"""

import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.device import best_device
from stemify.encoders.base import Encoder
from stemify.features import FrameFeatures, extract_features, stack_channels
from stemify.models.heads import CONTOUR_VOICING_THRESHOLD, decode_contour
from stemify.models.cuenet import CueNet
from stemify.pitch import PitchEstimator, PyinPitch
from stemify.sources import Sources


class LearnedEncoder(Encoder):
    """Neural cue producer: a ``CueNet``'s onset/voicing probs, plus a contour
    from its own pitch head when it has one and pyin when it does not."""

    name = "neural"

    def __init__(
        self,
        model: CueNet,
        device: str | None = None,
        pitch: PitchEstimator | None = None,
        use_pyin: bool = False,
    ) -> None:
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()
        self.pitch = pitch or PyinPitch()
        # A head on the model wins unless the caller insists otherwise. A checkpoint
        # carrying one was trained to produce the contour, and quietly preferring pyin
        # would report that run under a pitch source it never used.
        self.use_pyin = use_pyin

    @property
    def has_pitch_head(self) -> bool:
        """Whether the wrapped model produces its own contour."""
        return "pitch" in self.model.frequency_heads

    @property
    def pitch_source(self) -> str:
        """Which contour this encoder actually reports -- worth recording in a run."""
        return "pyin" if self.use_pyin or not self.has_pitch_head else "learned"

    @property
    def in_channels(self) -> int:
        """How many stacked renderings the wrapped model expects.

        A direct read. This was ``getattr(self.model, "in_channels", 1)``, defaulted so
        that a model which did not declare a width counted as single-channel -- but the
        only things that did not declare one were incomplete test doubles, and the
        default's cost fell on production: a 2-channel checkpoint whose width went
        unread would assemble a 1-channel input, and *1* is a plausible enough answer
        that the failure surfaces as a shape error in the first conv rather than as the
        misread it is. The doubles now declare the attribute instead.
        """
        return self.model.in_channels

    @torch.no_grad()
    def _forward(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """(onset probs, voicing probs, pitch logits) over a whole track's CQT.

        Voicing is ``None`` when the model has no voicing head, so an onset-only
        checkpoint still encodes (the decoder then falls back to the contour's
        voicing); pitch logits are ``None`` when it has no pitch head. Pitch is
        returned as raw logits, not probabilities: ``decode_bins`` normalises across
        the bin axis, so squashing each bin independently first would be the wrong
        distribution.

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
        pitch = None
        if "pitch" in out:
            pitch = out["pitch"].squeeze(0).cpu().numpy()   # (T, n_bins) logits
        return onset, voicing, pitch

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
        ``encode`` is this over a freshly computed CQT and pyin contour.

        When the model has a pitch head, the ``contour`` argument is *replaced* by the
        one the head produced rather than merged with it. Two contours cannot both be
        the pitch source, and the caller's is the cached pyin track this run exists to
        do without. ``use_pyin`` is how the caller asks for the other choice.
        """
        onset, voicing, pitch_logits = self._forward(features.values)
        if pitch_logits is not None and not self.use_pyin:
            contour = self.contour_from(pitch_logits, features.frame_times, voicing)
        return Cues(
            contour=contour,
            onset=onset,
            frame_times=features.frame_times,
            voicing=voicing,
            # Carried, not just consumed. The contour above is one *reading* of these
            # logits; a segmental decoder scores the distribution itself (path length,
            # spread), so collapsing to the contour here would silently zero those
            # potentials at decode while training saw them.
            pitch_logits=pitch_logits,
        )

    def contour_from(
        self,
        pitch_logits: np.ndarray,
        frame_times: np.ndarray,
        voicing: np.ndarray | None,
    ) -> Contour:
        """Per-bin pitch logits -> a ``Contour`` on the *feature* grid.

        Thin on purpose: the decode itself lives in ``models.heads.decode_contour``,
        which ``NeuralPitch`` also calls. Two entry points, one decode -- the property
        that makes "the learned contour" mean the same thing however it was reached.
        What this adds is only the operating point, which is this encoder's own
        (``CONTOUR_VOICING_THRESHOLD``) and not the decoder's.
        """
        voiced = None if voicing is None else voicing > CONTOUR_VOICING_THRESHOLD
        return decode_contour(pitch_logits, frame_times, voiced)
