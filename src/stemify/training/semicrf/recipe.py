"""``SemiCRFRecipe``: the structured NLL over whole tracks.

Simple and exact, and the reference the truncated-BPTT variant in ``semicrf.tbptt`` is
verified against. Its cost is the reason that variant exists: a track is ~7000 DP
frames, which forces B=2 and ~17 gradient steps/epoch on this corpus, and pads every
batch to its longest track (~40% waste).

The variant is a *subpackage* rather than a sibling because it is the same objective --
same potentials, same DP, same decoder -- differing only in how much of a track one
gradient step sees. Its recipe subclasses this one.
"""

from collections.abc import Sequence

import numpy as np
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.models.networks import tcn_receptive_field
from stemify.features import HOP_LENGTH, SAMPLE_RATE
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.semicrf import dataset as sequence_ds
from stemify.training.base import Loss
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.base import Recipe, _loader_kwargs
from stemify.training.frame.recipe import FrameRecipe


class SemiCRFRecipe(Recipe):
    """The structured NLL over whole tracks: encoder and segmental decoder, jointly.

    The decoder is held on the recipe so ``build_loss`` and ``decoder_from`` agree on
    its configuration (stride, dmax, duration edges); the *weights* still travel in
    the checkpoint, never on the recipe."""

    name = "semicrf"

    def __init__(
        self,
        conditions: Sequence[str] | None = None,
        decoder: SemiCRFDecoder | None = None,
        pitch: bool = False,
    ) -> None:
        if pitch:
            # Refused rather than ignored: the DP scores onset and voicing evidence
            # only, so a pitch branch here would train a head no potential reads and
            # report a successful run that learned nothing with it.
            raise NotImplementedError(
                "the semi-CRF scores onset and voicing evidence only, so a pitch "
                "branch would be trained but never read. Use --objective frame."
            )
        super().__init__(conditions)
        self.decoder = decoder or SemiCRFDecoder()

    def build_model(self) -> nn.Module:
        # Same encoder; only the objective differs.
        return FrameRecipe([c.name for c in self.conditions]).build_model()

    def build_loss(self, config: TrainConfig) -> Loss:
        return SemiCRFLoss(self.decoder)

    def _sequences(self, root: str, split: str, conditions, cache, limit: int | None):
        parts = [
            sequence_ds.SequenceDataset(c.dataset(root, split, limit), cache, c)
            for c in conditions
        ]
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    def train_loader(self, root, cache, config, limit) -> DataLoader:
        return DataLoader(
            self._sequences(root, "train", self.conditions, cache, limit), shuffle=True,
            collate_fn=sequence_ds.collate, **_loader_kwargs(config),
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        return DataLoader(
            self._sequences(root, "validation", [CONDITIONS["clean"]], cache, val_limit),
            collate_fn=sequence_ds.collate, **_loader_kwargs(config),
        )

    def decoder_from(self, state: dict) -> Decoder:
        decoder = SemiCRFDecoder(
            stride=self.decoder.stride,
            dmax_s=self.decoder.pot.dmax_s,
            dur_edges_s=self.decoder.pot.dur_edges_s.cpu().numpy(),
        )
        decoder.load_state_dict(state["loss"])
        return decoder
