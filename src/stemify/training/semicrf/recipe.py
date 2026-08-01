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

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from stemify.decoders.base import Decoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.training.base import Loss, Recipe, _loader_kwargs
from stemify.training.conditions import CONDITIONS
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.semicrf import dataset as sequence_ds
from stemify.training.semicrf.loss import SemiCRFLoss


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
        """``pitch`` adds the ``FrequencyHead``, whose per-bin logits the span
        features read. This recipe used to *refuse* the argument, on the reasoning
        that the DP scored onset and voicing only and a pitch branch would train a
        head no potential read. That stopped being true when the scorer gained its
        path and spread features -- both are functions of the head's distribution.

        """
        super().__init__(conditions)
        self.decoder = decoder or SemiCRFDecoder()
        self.pitch = pitch

    @property
    def heads(self) -> list[str]:
        """The binary pair, plus pitch when this run carries the head. Drives the
        cached targets and the checkpoint's head list, which must agree with the
        branches the model actually holds."""
        return [*super().heads, "pitch"] if self.pitch else super().heads

    @property
    def params(self) -> dict[str, object]:
        return {"pitch": self.pitch}

    def build_model(self) -> nn.Module:
        # Same encoder; only the objective differs.
        return FrameRecipe(
            [c.name for c in self.conditions], pitch=self.pitch
        ).build_model()

    def build_loss(self) -> Loss:
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
            dmax_s=self.decoder.scorer.dmax_s,
            dur_edges_s=self.decoder.scorer.dur_edges_s.cpu().numpy(),
        )
        decoder.load_state_dict(state["loss"])
        return decoder
