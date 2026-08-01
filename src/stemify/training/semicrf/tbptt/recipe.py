"""``SemiCRFTbpttRecipe``: the semi-CRF NLL over ordered chunks, carrying alpha.

The same objective as ``semicrf.SemiCRFRecipe`` -- by the telescoping identity the
per-chunk losses sum to the whole-track NLL exactly -- at ~13x the gradient steps for
the same DP work, with static shapes and no padding.
"""

import numpy as np
from torch.utils.data import DataLoader

from stemify.features import HOP_LENGTH, SAMPLE_RATE
from stemify.models.networks import tcn_receptive_field
from stemify.training.base import Loss
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.semicrf.recipe import SemiCRFRecipe
from stemify.training.semicrf.tbptt import dataset as chunked_ds
from stemify.training.semicrf.tbptt.loss import ChunkedSemiCRFLoss


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
