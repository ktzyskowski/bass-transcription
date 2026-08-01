"""``SemiCRFLoss``: the semi-CRF's structured NLL as a training objective.

The decoder-in-the-loop objective. Run the tagger over a padded batch of whole
tracks in one pass, wrap each track's onset/voicing head logits (sliced to its true
length) in a ``TensorCues``, and take the decoder's mean segmental NLL over the
batch. Grad flows back through the heads, so the ``Trainer``'s optimiser -- spanning
the model *and* this loss's ``parameters()`` (the decoder's potentials) -- trains the
encoder and the CRF jointly.

It pairs with ``SequenceDataset``/``collate`` (whole tracks, padded) exactly as
``FrameBCELoss`` pairs with ``FrameWindowDataset``: the batch shape and the objective
are the two things that vary between paradigms, and both plug into the one ``Trainer``.
"""

import torch
from torch import nn

from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.training.base import Loss
from stemify.training.semicrf.dataset import SequenceBatch


class SemiCRFLoss(Loss):
    """The semi-CRF NLL over whole tracks, training the encoder and decoder jointly."""

    def __init__(self, decoder: SemiCRFDecoder) -> None:
        self.decoder = decoder

    def parameters(self):
        # The decoder's potentials train with the encoder under one optimiser.
        return self.decoder.parameters()

    def state_dict(self):
        # The ~15 potentials ARE the decoder. Without them a checkpoint restores an
        # encoder paired with an untrained CRF -- w_on=1, dur=zeros, i.e. no duration
        # preference at all, which is precisely the mechanism this objective exists
        # to learn.
        return {k: v.detach().cpu().clone() for k, v in self.decoder.state_dict().items()}

    def load_state_dict(self, state):
        self.decoder.load_state_dict(state)

    def to(self, device: torch.device) -> "SemiCRFLoss":
        self.decoder.to(device)
        return self

    def compute(
        self, model: nn.Module, batch: SequenceBatch, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = model(batch.features.to(device))       # {head: (B, Tmax, 1)}
        onset = out["onset"].squeeze(-1)             # (B, Tmax)
        voicing = out["voicing"].squeeze(-1) if "voicing" in out else None
        # (B, T, K) -- the CRF's path/spread features read these.
        pitch = out["pitch"] if "pitch" in out else None

        # Slice each track to its true length (drop the right-padding) and wrap the
        # head logits as a grad-bearing TensorCues, so the batched NLL reaches the
        # heads. Contour/frame_times/notes are the per-track constants from collate.
        cues_list: list[TensorCues] = []
        notes_list = []
        for b, length in enumerate(batch.lengths):
            v = voicing[b, :length] if voicing is not None else None
            cues_list.append(
                TensorCues.from_logits(
                    batch.contours[b], onset[b, :length], batch.frame_times[b], v,
                    None if pitch is None else pitch[b, :length],
                )
            )
            notes_list.append(batch.notes[b])

        nll = self.decoder.nll_batch(cues_list, notes_list).mean()
        scalar = float(nll.item())
        return nll, {"nll": scalar, "total": scalar}

    def batch_size(self, batch: SequenceBatch) -> int:
        return len(batch)
