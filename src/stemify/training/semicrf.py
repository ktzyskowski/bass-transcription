"""End-to-end semi-CRF training: the forward that ties the encoder to the decoder.

The frame-head ``Trainer`` (``trainer.py``) fits per-frame BCE over fixed windows.
The semi-CRF trains a *different* objective over **whole tracks**: the decoder's
segmental negative log-likelihood, with the gradient flowing back through the frame
heads so the encoder learns to emit onsets/voicing the structured decoder can parse.

``crf_loss`` is the seam -- one track in, one differentiable loss out. It is kept a
plain function (no optimiser, no dataset) so the model->TensorCues->NLL path is
unit-testable with a fake tagger and a synthetic track, no audio. The training loop
composes it: sum ``crf_loss`` over a minibatch of tracks, ``backward``, step.

The tagger is fully convolutional over time, so a whole track goes through in one
pass -- there is no windowing here, because the DP needs the entire sequence at once
(a note can be longer than any window). That is the structural reason this cannot
reuse ``FrameWindowDataset`` and gets its own whole-track path.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch

from stemify.contour import Contour
from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.encoders.tagger import FrameTagger
from stemify.notes import NoteEvent


def crf_loss(
    model: FrameTagger,
    decoder: SemiCRFDecoder,
    features: np.ndarray,
    frame_times: np.ndarray,
    contour: Contour,
    notes: list[NoteEvent],
    device: torch.device | None = None,
) -> torch.Tensor:
    """The semi-CRF NLL for one whole track, differentiable through ``model``.

    Runs the tagger over the track's CQT in one pass, wraps the onset (and voicing,
    when present) head logits in a ``TensorCues`` -- grad-bearing, so the loss reaches
    the heads -- and scores the gold segmentation built from ``notes``. An optimiser
    step over ``model`` and ``decoder`` parameters trains the encoder and the CRF
    potentials jointly.

    ``contour`` (pyin) and ``frame_times`` are constants and stay on the CPU; the
    tagger and its logits move to ``device``. The decoder builds its own emission
    constants (the pitch-change cumsum) on the CPU, so a GPU loop must keep the whole
    step on one device -- reconciling that placement is the loop's job, deferred.
    """
    device = device or next(model.parameters()).device
    x = torch.as_tensor(np.ascontiguousarray(features)).float().to(device)
    out = model(x[None, None])  # (1, 1, n_bins, T) -> {head: (1, T, 1)}
    onset_logit = out["onset"].squeeze(0).squeeze(-1)  # (T,)
    voicing_logit = (
        out["voicing"].squeeze(0).squeeze(-1) if "voicing" in out else None
    )
    cues = TensorCues.from_logits(contour, onset_logit, frame_times, voicing_logit)
    return decoder.nll(cues, notes)


@dataclass
class CrfExample:
    """One whole track's inputs for CRF training: the CQT, its grid, the pyin
    contour and the reference notes. The audio-derived pieces (CQT, contour) are
    computed and cached once by the builder; this bundle is what a fit step consumes.
    """

    features: np.ndarray            # (n_bins, T) CQT
    frame_times: np.ndarray         # (T,)
    contour: Contour                # pyin pitch track
    notes: list[NoteEvent]          # reference notes -> gold segmentation


class SemiCRFTrainer:
    """Fit a ``FrameTagger`` + ``SemiCRFDecoder`` jointly on the segmental NLL.

    A separate loop from the frame-head ``Trainer``: that one serves fixed windows
    and sums per-frame BCE; this one iterates *whole tracks* and minimises the
    decoder's structured NLL, one optimiser over both the encoder's and the CRF's
    parameters. A minibatch loss is the mean per-track NLL, summed sequentially (the
    padded batched DP is a GPU-throughput optimisation, deferred). MLflow, validation
    note-F1 and checkpointing are the caller's to wire via ``on_epoch`` -- kept out so
    the loop stays unit-testable with fake tracks."""

    def __init__(
        self,
        model: FrameTagger,
        decoder: SemiCRFDecoder,
        learning_rate: float = 0.01,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.decoder = decoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            chain(self.model.parameters(), self.decoder.parameters()), lr=learning_rate
        )

    def loss_on(self, example: CrfExample) -> torch.Tensor:
        return crf_loss(
            self.model, self.decoder, example.features, example.frame_times,
            example.contour, example.notes, device=self.device,
        )

    def step(self, batch: Sequence[CrfExample]) -> float:
        """One optimiser step on a minibatch; returns the mean NLL."""
        self.optimizer.zero_grad()
        loss = torch.stack([self.loss_on(e) for e in batch]).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit(
        self,
        examples: Sequence[CrfExample],
        epochs: int = 10,
        batch_size: int = 8,
        shuffle: bool = True,
        on_epoch: Callable[[int, float], None] | None = None,
        seed: int = 0,
    ) -> list[float]:
        """Train for ``epochs`` over ``examples``; return the per-epoch mean NLL.

        ``on_epoch(epoch, mean_nll)`` is the hook for validation / logging /
        checkpointing -- called after each epoch with the model in its current state.
        """
        rng = random.Random(seed)
        order = list(range(len(examples)))
        history: list[float] = []
        for epoch in range(epochs):
            if shuffle:
                rng.shuffle(order)
            self.model.train()
            total, n = 0.0, 0
            for start in range(0, len(order), batch_size):
                batch = [examples[i] for i in order[start:start + batch_size]]
                total += self.step(batch) * len(batch)
                n += len(batch)
            mean = total / max(n, 1)
            history.append(mean)
            if on_epoch is not None:
                on_epoch(epoch, mean)
        return history
