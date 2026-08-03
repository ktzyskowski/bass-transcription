"""``SemiCRFRecipe``: the structured NLL, trained over ordered chunks of a track.

There were briefly two recipes here -- one feeding whole padded tracks, one feeding
fixed chunks -- computing the same objective through two datasets, two batch types and
two losses. They differed on exactly one axis, whether the alpha carry is detached
between chunks, so that is now a flag on ``SemiCRFLoss`` and this is the only recipe.

The *decoder* stays in ``decoders.semicrf``. This package is how the thing is trained;
that one is how it decodes at inference.
"""

from collections.abc import Sequence

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from stemify.decoders.base import Decoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.features import HOP_LENGTH, SAMPLE_RATE
from stemify.models.networks import tcn_receptive_field
from stemify.training.base import Loss, Recipe
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.semicrf import dataset as chunked_ds
from stemify.training.semicrf.loss import SemiCRFLoss

# Chunk length in DP frames. ~4.5 s at the decimated hop -- comfortably longer than
# dmax (115 frames, 4 s), so most spans are scored entirely inside one chunk and only
# a boundary-crossing minority relies on the carry.
CHUNK_FRAMES = 128


class SemiCRFRecipe(Recipe):
    """The structured NLL: encoder and segmental decoder, trained jointly.

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
    def config(self) -> dict[str, object]:
        """Everything ``__init__`` takes -- including the decoder's *geometry*.

        Stride and dmax are not hyperparameters that a default can stand in for: they
        set the DP's grid, so a checkpoint rebuilt on the wrong ones is scored over
        spans of the wrong length. They were absent from the checkpoint entirely until
        this existed, which was harmless only because nothing had varied them yet.
        """
        return {
            **super().config,
            "pitch": self.pitch,
            "decoder": {
                "stride": self.decoder.stride,
                "dmax_s": self.decoder.scorer.dmax_s,
                "dur_edges_s": self.decoder.scorer.dur_edges_s.cpu().tolist(),
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SemiCRFRecipe":
        """The decoder is an object, so it is revived rather than splatted."""
        config = dict(config)
        geometry = dict(config.pop("decoder", {}))
        decoder = SemiCRFDecoder(
            stride=geometry.get("stride", SemiCRFDecoder().stride),
            **{
                k: (np.asarray(v, dtype=np.float32) if k == "dur_edges_s" else v)
                for k, v in geometry.items() if k in ("dmax_s", "dur_edges_s")
            },
        )
        return cls(decoder=decoder, **config)

    @property
    def params(self) -> dict[str, object]:
        return {"pitch": self.pitch, "chunk_frames": CHUNK_FRAMES}

    def build_model(self) -> nn.Module:
        # Same encoder; only the objective differs.
        return FrameRecipe(
            [c.name for c in self.conditions], pitch=self.pitch
        ).build_model()

    def build_loss(self) -> Loss:
        return SemiCRFLoss(self.decoder)

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
        dmax = self.decoder.scorer.grid(np.arange(2) * dp_hop).dmax
        half = tcn_receptive_field() // 2
        right = -(-half // self.decoder.stride)     # ceil, into DP frames
        return dmax, right

    def _chunks(self, root: str, split: str, conditions, cache, config, limit):
        dmax, right = self._geometry
        parts = [
            chunked_ds.ChunkDataset(
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

    def decoder_from(self, state: dict) -> Decoder:
        decoder = SemiCRFDecoder(
            stride=self.decoder.stride,
            dmax_s=self.decoder.scorer.dmax_s,
            dur_edges_s=self.decoder.scorer.dur_edges_s.cpu().numpy(),
        )
        # A checkpoint from before the pyin span features were removed carries
        # `scorer.w_pitch`. Loading it strictly fails, which is right -- the scorer it
        # was trained against no longer exists, so those potentials describe a
        # different model -- but the bare key error does not say that.
        retired = sorted(k for k in state["loss"] if k.endswith("w_pitch"))
        if retired:
            raise ValueError(
                f"checkpoint carries retired potential(s) {retired}: it was trained "
                "with the pyin interior-jump span feature, which the scorer no longer "
                "has. Its weights are not comparable to a current run -- retrain, or "
                "score it with the revision that produced it."
            )
        decoder.load_state_dict(state["loss"])
        return decoder
