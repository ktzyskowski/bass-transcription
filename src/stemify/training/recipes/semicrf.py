"""The semi-CRF recipes: whole-track NLL, and its truncated-BPTT form.

Two recipes for one objective, differing only in how much of a track a gradient step
sees:

  - ``SemiCRFRecipe``      -- whole tracks. Simple and exact, but a track is ~7000 DP
    frames, which forces B=2 and ~17 gradient steps/epoch on this corpus, and pads
    every batch to its longest track (~40% waste).
  - ``SemiCRFTbpttRecipe`` -- fixed chunks with the alpha window carried between them.
    The same objective by the telescoping identity, at ~13x the gradient steps for the
    same DP work, with static shapes and no padding.

They are separate recipes rather than a flag because the batch shape, the loss and the
loader all change together -- which is exactly the coupling ``Recipe`` exists to bind.
"""

from collections.abc import Sequence

import numpy as np
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.encoders.networks import tcn_receptive_field
from stemify.features import HOP_LENGTH, SAMPLE_RATE
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.datasets import chunked as chunked_ds
from stemify.training.datasets import sequence as sequence_ds
from stemify.training.losses import Loss
from stemify.training.losses.segmental import SemiCRFLoss
from stemify.training.losses.tbptt import ChunkedSemiCRFLoss
from stemify.training.recipes.base import Recipe, _loader_kwargs
from stemify.training.recipes.frame import FrameRecipe


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
    ) -> None:
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

# Chunk length in DP frames. ~4.5 s at the decimated hop -- comfortably longer than
# dmax (115 frames, 4 s), so most spans are scored entirely inside one chunk and only
# a boundary-crossing minority relies on the carry.
CHUNK_FRAMES = 128


class SemiCRFTbpttRecipe(SemiCRFRecipe):
    """The semi-CRF NLL over ordered chunks, carrying alpha between them.

    Same objective and same decoder as ``SemiCRFRecipe`` -- by the telescoping identity
    the per-chunk losses sum to the whole-track NLL exactly -- so the two differ only in
    how much of a track one gradient step sees, which is the whole point: whole tracks
    give ~17 steps/epoch here, chunks give ~13x that for the same DP work.

    Subclassing rather than duplicating: ``build_model``, ``decoder_from`` and the
    conditions handling are identical, and only the loader and the loss change. If a
    third variant ever wants a different encoder this should flatten back out.
    """

    name = "semicrf-tbptt"

    def build_loss(self, config: TrainConfig) -> Loss:
        return ChunkedSemiCRFLoss(self.decoder)

    @property
    def _geometry(self) -> tuple[int, int]:
        """``(dmax, right_context)`` in DP frames.

        ``dmax`` comes from the decoder's own ``grid`` over a synthetic two-frame
        stretch of the DP hop, rather than being recomputed here, so the dataset's
        left context and the scan's candidate-start span cannot drift apart.

        ``right_context`` is half the encoder's receptive field: an output frame at the
        chunk's right edge reads that far into the future, and without the margin it
        would read zero-padding instead. The DP does not scan it -- see
        ``forward.chunk_log_partition``'s ``right``.
        """
        dp_hop = HOP_LENGTH / SAMPLE_RATE * self.decoder.stride
        dmax, _ = self.decoder.pot.grid(np.arange(2) * dp_hop)
        half = tcn_receptive_field() // 2
        right = -(-half // self.decoder.stride)     # ceil, into DP frames
        return dmax, right

    def _chunks(self, root: str, split: str, conditions, cache, config, limit):
        dmax, right = self._geometry
        parts = [
            chunked_ds.ChunkedSequenceDataset(
                c.dataset(root, split, limit), cache, c,
                chunk_frames=CHUNK_FRAMES, dmax=dmax, stride=self.decoder.stride,
                batch_size=config.batch_size, right_context=right,
            )
            for c in conditions
        ]
        if len(parts) != 1:
            raise ValueError(
                "chunked training runs one condition at a time: streams from different "
                "conditions would interleave in a batch and share a carry keyspace"
            )
        return parts[0]

    def _loader(self, dataset) -> DataLoader:
        # batch_size=None: the dataset emits whole ChunkBatch objects, because chunks
        # must arrive in per-stream order and a sampler cannot express that. Workers
        # each own a disjoint slice of tracks and stamp globally unique stream ids.
        return DataLoader(
            dataset, batch_size=None, num_workers=0, collate_fn=None,
        )

    def train_loader(self, root, cache, config, limit) -> DataLoader:
        return self._loader(
            self._chunks(root, "train", self.conditions, cache, config, limit)
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        return self._loader(
            self._chunks(root, "validation", [condition], cache, config, val_limit)
        )
