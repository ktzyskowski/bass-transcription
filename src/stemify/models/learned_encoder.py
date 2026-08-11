"""``LearnedEncoder``: trained backbone + heads as the cue producer.

Runs a ``FrameNet`` over a track's CQT to produce per-frame onset, voicing and pitch
in one pass, and returns a ``Frames``. The model is fully convolutional over time, so
the whole track goes through at once -- no windowing at inference.

**Pitch comes from the model, and only from the model.** There used to be a second
route: an injected ``PitchEstimator`` (pyin) that a checkpoint without a
``FrequencyHead`` fell back to, plus a ``use_pyin`` flag to force it on a checkpoint
that had one. That fallback outlived its purpose -- the learned head shipped at +5.0
strict F1 and broke the recall ceiling pyin imposed, and every checkpoint trained since
2026-07-31 carries one. What remained was a branch, a flag, a ``pitch_source``
property, an ABC with one real implementation, and a second contour cache, all to
serve a path no run took.

A model with no pitch head still encodes: its ``Frames`` carry onset and voicing and
no contour, which is exactly what cues-level scoring wants and what ``GreedyDecoder``
refuses by name. That is the honest failure -- a decoder that names notes by pitch
cannot name them from a model that does not produce it.

Lives here rather than in ``transcription`` because it holds a ``FrameNet``:
``Encoder`` is the contract and this is its implementation, which is what keeps the
contract layer free of torch.
"""

from dataclasses import replace

import numpy as np
import torch

from stemify.music.contour import Contour
from stemify.transcription.frames import Frames
from stemify.platform.device import best_device
from stemify.transcription.encoder import Encoder
from stemify.audio.features import FrameFeatures, extract_features, stack_channels
from stemify.models.heads import CONTOUR_VOICING_THRESHOLD, decode_contour
from stemify.models.framenet import FrameNet
from stemify.models.frames import frames_from_output, to_numpy
from stemify.audio.sources import Sources


class LearnedEncoder(Encoder):
    """Neural cue producer: a ``FrameNet``'s onset, voicing and pitch, in one pass."""

    name = "neural"

    def __init__(self, model: FrameNet, device: str | None = None) -> None:
        self.device = torch.device(device or best_device())
        self.model = model.to(self.device).eval()

    @property
    def has_pitch_head(self) -> bool:
        """Whether the wrapped model produces a contour. False for the checkpoints
        predating the pitch head, which can still be scored at cues level (onset and
        voicing) but can no longer be decoded to notes."""
        return "pitch" in self.model.frequency_heads

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
        return self.encode_features(stack_channels(channels))

    def encode_features(self, features: FrameFeatures) -> Frames:
        """``Frames`` from a precomputed CQT -- the cache-friendly path the async
        validator uses (no CQT recompute). ``encode`` is this over a fresh one.

        It used to take a ``contour`` argument, for the model that had no pitch head to
        fall back on. Nothing supplies one now, so there is no argument and no branch:
        the contour is the head's or there is none.
        """
        frames = self._forward(features.values, features.frame_times)
        pitch_logits, voicing = frames.pitch_logits, frames.voicing
        contour = None
        if pitch_logits is not None:
            contour = self.contour_from(
                np.asarray(pitch_logits),
                features.frame_times,
                None if voicing is None else np.asarray(voicing),
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

        Thin on purpose: the decode itself lives in ``models.heads.decode_contour``.
        What this adds is only the operating point, which is this encoder's own
        (``CONTOUR_VOICING_THRESHOLD``) and not the decoder's.
        """
        voiced = None if voicing is None else voicing > CONTOUR_VOICING_THRESHOLD
        return decode_contour(pitch_logits, frame_times, voiced)
