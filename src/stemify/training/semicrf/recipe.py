"""``SemiCRFRecipe``: the structured NLL, trained over ordered chunks of a track.

There were briefly two recipes here -- one feeding whole padded tracks, one feeding
fixed chunks -- computing the same objective through two datasets, two batch types and
two losses. They differed on exactly one axis, whether the alpha carry is detached
between chunks, so that is now a flag on ``SemiCRFLoss`` and this is the only recipe.

The *decoder* stays in ``decoders.semicrf``. This package is how the thing is trained;
that one is how it decodes at inference.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

import torch

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.decoders.semicrf.constants import DMAX_S, DUR_EDGES_S, STRIDE
from stemify.decoders.semicrf.spans import LinearSpanScorer
from stemify.features import HOP_LENGTH, SAMPLE_RATE
from stemify.models.networks import tcn_receptive_field
from stemify.models.cuenet import CueNet
from stemify.training.base import Loss, Recipe, config_field, config_str_list
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS, Condition
from stemify.training.config import TrainConfig
from stemify.training.frame.recipe import FrameRecipe
from stemify.training.semicrf import dataset as chunked_ds
from stemify.training.semicrf.dataset import ChunkBatch
from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.semicrf.tracks import TrackInputs, track_inputs

# Chunk length in DP frames. ~4.5 s at the decimated hop -- comfortably longer than
# dmax (115 frames, 4 s), so most spans are scored entirely inside one chunk and only
# a boundary-crossing minority relies on the carry.
CHUNK_FRAMES = 128

# Weight on the auxiliary pitch cross-entropy in `NLL + w * pitch_ce`. 1.0 is a
# starting point, not a tuned value: `pitch_ce` opens near ln(216) ~ 5.4 while the
# per-chunk NLL is on a scale set by the track and the potentials, so the two are not
# guaranteed comparable and this is the first knob to sweep. Recorded in `config` so a
# checkpoint says which value produced it.
PITCH_WEIGHT = 1.0


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
        pitch_weight: float = PITCH_WEIGHT,
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
        self.pitch_weight = pitch_weight

    @property
    def heads(self) -> list[str]:
        """The binary pair, plus pitch when this run carries the head. Drives the
        cached targets and the checkpoint's head list, which must agree with the
        branches the model actually holds."""
        return [*super().heads, "pitch"] if self.pitch else super().heads

    def _serialisable_scorer(self) -> LinearSpanScorer:
        """This recipe's scorer, narrowed to the one shape a checkpoint can rebuild.

        Duration-bucket edges are a ``LinearSpanScorer`` parameterisation, not part of
        the ``SpanScorer`` contract -- ``dmax_s`` is the geometry every scorer has, and
        it lives on the ABC. So recording the edges means asserting the scorer is
        linear, and the assertion belongs here, once and loudly.

        A neural scorer will have its own configuration to serialise. Raising is what
        keeps that an explicit gap rather than a checkpoint that records linear
        geometry for a scorer that never had any, and rebuilds as the wrong model.
        """
        scorer = self.decoder.scorer
        if not isinstance(scorer, LinearSpanScorer):
            raise NotImplementedError(
                f"{type(scorer).__name__} has no checkpoint representation yet: only "
                "LinearSpanScorer's duration-bucket edges are serialised. Give the "
                "scorer its own config before training a run you intend to reload."
            )
        return scorer

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
            "pitch_weight": self.pitch_weight,
            "decoder": {
                "stride": self.decoder.stride,
                "dmax_s": self.decoder.scorer.dmax_s,
                "dur_edges_s": self._serialisable_scorer().dur_edges.cpu().tolist(),
            },
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "SemiCRFRecipe":
        """Field by field, because the decoder's geometry is an object to revive and
        the rest arrives as plain data off disk. A wrong type fails here, naming the
        field, rather than as a shape error inside the scan."""
        geometry = config.get("decoder") or {}
        if not isinstance(geometry, Mapping):
            raise ValueError(
                f"recipe config field 'decoder' should be a mapping of geometry, got "
                f"{type(geometry).__name__}"
            )
        # The constructor's own defaults, read from the constants rather than off a
        # throwaway ``SemiCRFDecoder()`` built only to be measured. Same values, one
        # fewer object -- and it no longer assumes the default decoder's scorer is the
        # linear one just to recover a number that never depended on the scorer.
        edges = geometry.get("dur_edges_s")
        decoder = SemiCRFDecoder(
            stride=config_field(geometry, "stride", int, STRIDE),
            dmax_s=config_field(geometry, "dmax_s", float, DMAX_S),
            dur_edges_s=(
                DUR_EDGES_S if edges is None
                else np.asarray(edges, dtype=np.float32)
            ),
        )
        return cls(
            conditions=config_str_list(config, "conditions"),
            decoder=decoder,
            pitch=config_field(config, "pitch", bool, False),
            pitch_weight=config_field(config, "pitch_weight", float, PITCH_WEIGHT),
        )

    @property
    def aux_heads(self) -> list[str]:
        """Heads needing a *per-frame* target, so the dataset caches and serves one.

        Deliberately not ``self.heads``. The structured NLL supervises onset and
        voicing through the segmentation itself -- their frame targets would be
        redundant. Pitch is the exception: the span features read how far the
        distribution *moves* and how much it disagrees, never whether it is right, so
        nothing in the NLL constrains the head's values and an auxiliary term has to.
        """
        return ["pitch"] if self.pitch else []

    @property
    def params(self) -> dict[str, object]:
        return {
            "pitch": self.pitch,
            "chunk_frames": CHUNK_FRAMES,
            "pitch_weight": self.pitch_weight,
        }

    def build_model(self) -> CueNet:
        # Same encoder; only the objective differs.
        return FrameRecipe(
            [c.name for c in self.conditions], pitch=self.pitch
        ).build_model()

    def build_loss(self) -> Loss[chunked_ds.ChunkBatch]:
        return SemiCRFLoss(
            self.decoder, aux_heads=self.aux_heads, aux_weight=self.pitch_weight
        )

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

    def _chunks(
        self,
        root: str,
        split: str,
        conditions: Sequence[Condition],
        cache: Cache,
        config: TrainConfig,
        limit: int | None,
    ) -> chunked_ds.ChunkDataset:
        """One dataset over the union of ``conditions`` -- domain randomisation.

        This used to build one ``ChunkDataset`` per condition and then refuse any set
        but a singleton, on the grounds that "streams from different conditions would
        interleave in a batch and share a carry keyspace". The keyspace collision was
        real *for that shape*: each dataset numbered its streams from the same base, so
        stream 0 of one condition and stream 0 of another indexed the same carry. It is
        not a property of mixing conditions. One dataset over the concatenated items has
        a single keyspace and no collision, and the interleaving it warned about is
        precisely what domain randomisation is.

        Nothing below ``track_inputs`` knows a track's condition -- a ``TrackInputs`` is
        feature paths plus notes -- and ``resolve()`` has already rejected a mixed-width
        set, so the windows stack identically. The frame path has always trained on the
        union this way; the structured objective was the odd one out, which made any
        comparison against a frame baseline vary two things at once.
        """
        dmax, right = self._geometry
        items: list[TrackInputs] = []
        for c in conditions:
            items += track_inputs(
                c.dataset(root, split, limit), c, cache, heads=self.aux_heads
            )
        return chunked_ds.ChunkDataset(
            items,
            chunk_frames=CHUNK_FRAMES, dmax=dmax, stride=self.decoder.stride,
            batch_size=config.batch_size, right_context=right,
            heads=self.aux_heads,
        )

    def _loader(self, dataset: chunked_ds.ChunkDataset) -> DataLoader[ChunkBatch]:
        # batch_size=None: the dataset emits whole ChunkBatch objects, because chunks
        # must arrive in per-stream order and a sampler cannot express that. Workers
        # each own a disjoint slice of tracks and stamp globally unique stream ids.
        return DataLoader(
            dataset, batch_size=None, num_workers=0, collate_fn=None,
        )

    def train_loader(
        self, root: str, cache: Cache, config: TrainConfig, limit: int | None
    ) -> DataLoader[ChunkBatch]:
        return self._loader(
            self._chunks(root, "train", self.conditions, cache, config, limit)
        )

    def val_loader(
        self, root: str, cache: Cache, config: TrainConfig, val_limit: int | None
    ) -> DataLoader[ChunkBatch]:
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        return self._loader(
            self._chunks(root, "validation", [condition], cache, config, val_limit)
        )

    def decoder_from(self, state: Mapping[str, Any]) -> Decoder:
        decoder = SemiCRFDecoder(
            stride=self.decoder.stride,
            dmax_s=self.decoder.scorer.dmax_s,
            dur_edges_s=self._serialisable_scorer().dur_edges.cpu().numpy(),
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
