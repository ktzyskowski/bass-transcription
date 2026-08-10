"""``LearnedEncoder``: trained backbone + heads as the cue producer.

Runs a ``CueNet`` over a track's CQT to produce per-frame onset (and voicing,
when the model has that head) probabilities, pairs them with a pitch contour, and
returns a ``Frames``. The model is fully convolutional over time, so the whole track
goes through in one pass -- no windowing at inference.

Pitch comes from one of two places. A model carrying a ``FrequencyHead`` produces it
in the *same forward pass* as onset and voicing; otherwise it falls back to the
injected ``PitchEstimator`` (``PyinPitch`` by default).

The learned head *does* also exist as a ``PitchEstimator`` -- ``NeuralPitch``, its
sibling in this package -- for callers holding only a path. This class does not use it, for one reason: by the
time it wants a contour it has already computed the CQT and run the backbone, so going
back out through ``estimate(audio_path)`` would pay for both again. It reads the pitch
logits out of the forward pass it just did instead.

That is an optimisation, not a second implementation. Both routes end in
``models.heads.decode_contour``, so "the learned contour" means the same thing however
it was reached -- which is what makes the two substitutable rather than merely similar.

``use_pyin=True`` forces the DSP contour even on a model that has a head, which is how
an A/B between the two is run.

Lives here rather than beside ``ClassicalEncoder`` in ``transcription.encoders`` because it holds
a ``CueNet``: ``Encoder`` is the contract, this is its neural implementation, and the
split is what keeps the DSP path free of torch.
"""

from dataclasses import replace

import numpy as np
import torch

from stemify.music.contour import Contour
from stemify.transcription.frames import Frames
from stemify.platform.device import best_device
from stemify.transcription.encoders.base import Encoder
from stemify.audio.features import FrameFeatures, extract_features, stack_channels
from stemify.models.heads import CONTOUR_VOICING_THRESHOLD, decode_contour
from stemify.models.cuenet import CueNet
from stemify.models.frames import frames_from_output, to_numpy
from stemify.transcription.pitch import PitchEstimator, PyinPitch
from stemify.audio.sources import Sources


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
    def _forward(self, features: np.ndarray, frame_times: np.ndarray) -> Frames:
        """A whole track's CQT -> numpy ``Frames``, contour not yet attached.

        Whichever heads the checkpoint carries, and no others: an onset-only model still
        encodes, and ``"pitch" in frames.heads`` is how a caller asks whether this one
        produces its own contour. The squashing and the trailing-axis handling are
        ``models.frames``' job, so this path and the training path cannot disagree about
        what an onset value *is*.

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
        out, hidden = self.model.heads_and_hidden(x[None].to(self.device))
        (frames,) = frames_from_output(out, [frame_times], hidden=hidden)
        return to_numpy(frames)

    def encode(self, sources: Sources) -> Frames:
        """``Frames`` for a track, computing whatever the model's input needs.

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

    def encode_features(
        self, contour: Contour | None, features: FrameFeatures
    ) -> Frames:
        """``Frames`` from a precomputed CQT, and a contour only if this encoder needs
        one -- the cache-friendly path the async validator uses (no CQT recompute).
        ``encode`` is this over a freshly computed CQT.

        When the model has a pitch head, the ``contour`` argument is *replaced* by the
        one the head produced rather than merged with it. Two contours cannot both be
        the pitch source. ``use_pyin`` is how the caller asks for the other choice.

        ``None`` is therefore allowed exactly when the head will supply it, and
        ``pitch_source`` is the property that says which case a caller is in. Passing
        ``None`` to an encoder that needs one raises rather than defaulting: a silent
        fallback here is how a run reports numbers under a pitch source nobody chose,
        and the difference is worth ~5 F1.
        """
        frames = self._forward(features.values, features.frame_times)
        pitch_logits, voicing = frames.pitch_logits, frames.voicing
        if pitch_logits is not None and not self.use_pyin:
            contour = self.contour_from(
                np.asarray(pitch_logits),
                features.frame_times,
                None if voicing is None else np.asarray(voicing),
            )
        elif contour is None:
            raise ValueError(
                f"this encoder's pitch source is {self.pitch_source!r}, so "
                "encode_features needs a contour: the model has no pitch head "
                "(or use_pyin was set), and features alone cannot give pitch."
            )
        # Every head rides along untouched, pitch logits included. The contour above is
        # one *reading* of those logits; a segmental decoder scores the distribution
        # itself (path length, spread, per-semitone log-probability), so collapsing to
        # the contour here would silently zero those potentials at decode while
        # training saw them.
        return replace(frames, contour=contour)

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
