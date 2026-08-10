"""``SemiCRFRecipe``: the structured NLL, trained over ordered chunks of a track.

There were briefly two recipes here -- one feeding whole padded tracks, one feeding
fixed chunks -- computing the same objective through two datasets, two batch types and
two losses. They differed on exactly one axis, whether the alpha carry is detached
between chunks, so that is now a flag on ``SemiCRFLoss`` and this is the only recipe.

The *decoder* is the parent package. This subpackage is how the thing is trained; the
parent is how it decodes at inference.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


from stemify.persistence.store import Store
from stemify.transcription.decoders.base import Decoder
from stemify.semicrf import SemiCRFDecoder
from stemify.semicrf.constants import DMAX_S, STRIDE
from stemify.semicrf.build import build_scorer
from stemify.semicrf.labelset import LabelSet
from stemify.semicrf.neural import NeuralSpanScorer
from stemify.semicrf.spans import SpanScorer
from stemify.audio.features import HOP_LENGTH, SAMPLE_RATE
from stemify.models.networks import TCN, tcn_receptive_field
from stemify.models.framenet import FrameNet
from stemify.fitting.base import (
    Loss, Recipe, config_field, config_int_list, config_str_list,
)
from stemify.core.conditions import CLEAN_BY_WIDTH, CONDITIONS, Condition
from stemify.fitting.config import TrainConfig
from stemify.models.assembly import build_framenet
from stemify.semicrf.fit import dataset as chunked_ds
from stemify.semicrf.fit.dataset import ChunkBatch
from stemify.semicrf.fit.loss import SemiCRFLoss
from stemify.semicrf.fit.tracks import TrainItem, train_items

# Chunk length in DP frames. ~4.5 s at the decimated hop -- comfortably longer than
# dmax (115 frames, 4 s), so most spans are scored entirely inside one chunk and only
# a boundary-crossing minority relies on the carry.
CHUNK_FRAMES = 128

# Weights on the auxiliary per-frame terms in `NLL/frames + sum_h w_h * term_h`.
# Both default to 1.0, which means something now that the NLL is a per-frame rate: all
# three terms are nats about the same 11.6 ms of audio, so a weight is a ratio of
# information rates rather than a number absorbing a frame count. Still starting points
# rather than tuned values -- `pitch_ce` opens near ln(216) ~ 5.4 against BCE's ln 2 ~
# 0.69 -- and both are recorded in `config` so a checkpoint says what produced it.
PITCH_WEIGHT = 1.0

# 0 disables the corresponding term, reproducing the structured-only objective. Both
# binary heads need one, and the measurement is unambiguous -- the same validator over
# the same targets, frame BCE against semi-CRF:
#
#     onset AP        frame 0.994-0.996   semi-CRF 0.20-0.53
#     voicing AUROC   frame 0.995-0.996   semi-CRF 0.510-0.522
#
# across five frame runs and three semi-CRF runs, so it is structural rather than an
# early-epoch artefact. The NLL is discriminative over segmentations: a feature earns
# gradient only in proportion to how much it helps rank gold above the alternatives,
# and every feature carries a learned scalar that can shrink toward zero. Nothing in
# the objective asks a head to be *calibrated*, only to be useful for ranking.
VOICING_WEIGHT = 1.0
ONSET_WEIGHT = 1.0


def _config_weights(config: Mapping[str, object], key: str) -> dict[str, float]:
    """A ``{head: float}`` field off a checkpoint's config, checked.

    Its own helper rather than ``config_field(..., dict, {})`` for the reason
    ``config_str_list`` exists: ``isinstance(x, dict)`` says nothing about what is
    *in* it, and this arrives as plain data off disk written by an older version.
    """
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"recipe config field {key!r} should be a mapping of head to weight, got "
            f"{type(value).__name__}"
        )
    bad = [k for k, v in value.items() if not isinstance(k, str)
           or isinstance(v, bool) or not isinstance(v, (int, float))]
    if bad:
        raise ValueError(
            f"recipe config field {key!r} has non-(str, number) entries: {bad!r}"
        )
    return {str(k): float(v) for k, v in value.items()}


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
        voicing_weight: float = VOICING_WEIGHT,
        onset_weight: float = ONSET_WEIGHT,
        pos_weight: Mapping[str, float] | None = None,
        scorer: str = "linear",
        harmonics: Sequence[int] | None = None,
        segment_chunks: int = 0,
    ) -> None:
        """``pitch`` adds the ``FrequencyHead``, whose per-bin logits the span
        features read. This recipe used to *refuse* the argument, on the reasoning
        that the DP scored onset and voicing only and a pitch branch would train a
        head no potential read. That stopped being true when the scorer gained its
        path and spread features -- both are functions of the head's distribution.

        ``segment_chunks`` cuts each track into independent chains of that many chunks
        and carries alpha *undetached* within one, which is what makes the gradient the
        objective's own. ``0`` keeps the whole-track carry and the detached, truncated
        gradient every earlier run used -- kept as the default so a checkpoint or a
        replay of an old command reproduces what it did before, not silently better.

        The chain length is the whole trade. Backprop through an undetached carry means
        one optimiser step per chain, so short chains buy steps and bounded activation
        memory while long ones buy an objective closer to the whole track's. Four is
        ~18 s at the pinned hop -- several times the longest representable segment, so
        the conditioning at each chain's start reaches nothing that matters.
        """
        super().__init__(conditions)
        # The *encoder's* configuration, and it belongs to the encoder rather than to
        # either objective: harmonic stacking is what the front-end sees, and the
        # structured NLL sits several stages downstream of it. This recipe used to
        # drop the argument and build at `(1,)` whatever the caller asked for, which
        # made "semi-CRF" and "HCQT" mutually exclusive for no reason either one
        # could state.
        self.harmonics = tuple(harmonics or (1,))
        # An explicit decoder wins: `from_config` rebuilds one whole, and a caller
        # passing both would otherwise get the string's scorer silently.
        self.decoder = decoder or SemiCRFDecoder(
            scorer=None if scorer == "linear"
            else self._neural_scorer(scorer, pitch, self.harmonics, self.in_channels)
        )
        self.pitch = pitch
        self.pitch_weight = pitch_weight
        self.voicing_weight = voicing_weight
        self.onset_weight = onset_weight
        self.pos_weight = dict(pos_weight or {})
        self.segment_chunks = segment_chunks

    @staticmethod
    def _neural_scorer(
        kind: str, pitch: bool, harmonics: Sequence[int], in_channels: int
    ) -> SpanScorer:
        """A ``NeuralSpanScorer`` sized to the backbone this recipe will build.

        Both widths come off a throwaway ``TCN`` rather than from constants: the
        trunk width is what ``phi`` projects from, and the pitch axis is what the
        alphabet is derived from (``LabelSet.from_bins``), so a change to either --
        harmonic stacking crops the axis, for one -- has to reach the scorer or the
        two disagree about how many semitones exist.
        """
        if kind != "neural":
            raise ValueError(f"unknown scorer {kind!r}; have 'linear' and 'neural'")
        if not pitch:
            raise ValueError(
                "--scorer neural needs --pitch. Its alphabet names semitones and "
                "every pitch label is scored against the pitch head's own "
                "per-semitone log-probability; without that head there is no "
                "pitch-valued evidence at all, and the 73 labels would score "
                "identically -- log Z spread over every relabelling of one "
                "segmentation, which is strictly worse than {note, rest}."
            )
        # A throwaway backbone, discarded after two reads. Cheaper than the drift:
        # both numbers are the model's, and a constant repeated here would be wrong
        # the first time the frontend's crop or the trunk's width moved -- harmonic
        # stacking crops the fundamental axis from 216 bins to 144, which is the
        # difference between a 72-semitone alphabet and a 48-semitone one.
        backbone = TCN(in_channels=in_channels, harmonics=harmonics)
        return NeuralSpanScorer(
            labels=LabelSet.from_bins(backbone.frontend.n_bins),
            trunk_dim=backbone.hidden,
            # `_geometry` derives the same dmax from the scorer's own `grid`, and
            # this scorer's `grid` raises when the two disagree -- so a mistake here
            # surfaces at the first batch rather than as a mis-sized table.
            dmax_frames=round(DMAX_S / (HOP_LENGTH / SAMPLE_RATE * STRIDE)),
        )

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
            "harmonics": list(self.harmonics),
            "pitch_weight": self.pitch_weight,
            "voicing_weight": self.voicing_weight,
            "onset_weight": self.onset_weight,
            "pos_weight": self.pos_weight,
            "segment_chunks": self.segment_chunks,
            "decoder": {
                "stride": self.decoder.stride,
                "dmax_s": self.decoder.scorer.dmax_s,
                "scorer": self.decoder.scorer.config,
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
        # A checkpoint predating `SpanScorer.config` records the linear scorer's
        # duration edges directly under `decoder`. Reading that shape as a linear
        # config keeps those checkpoints loadable, and is why `kind` defaults.
        scorer_config = geometry.get("scorer")
        if not isinstance(scorer_config, Mapping):
            scorer_config = {
                "kind": "linear",
                "dmax_s": config_field(geometry, "dmax_s", float, DMAX_S),
                "dur_edges_s": geometry.get("dur_edges_s"),
            }
        decoder = SemiCRFDecoder(
            stride=config_field(geometry, "stride", int, STRIDE),
            scorer=build_scorer(scorer_config),
        )
        return cls(
            conditions=config_str_list(config, "conditions"),
            decoder=decoder,
            harmonics=[int(h) for h in config_int_list(config, "harmonics")] or None,
            pitch=config_field(config, "pitch", bool, False),
            pitch_weight=config_field(config, "pitch_weight", float, PITCH_WEIGHT),
            # Defaults to 0 rather than VOICING_WEIGHT when absent: a checkpoint written
            # before this existed was trained *without* the term, and rebuilding it with
            # one would misreport how it was trained.
            voicing_weight=config_field(config, "voicing_weight", float, 0.0),
            onset_weight=config_field(config, "onset_weight", float, 0.0),
            pos_weight=_config_weights(config, "pos_weight"),
            # 0 for a checkpoint written before segments existed, which is what those
            # runs did: one carry per track, detached, truncated gradient.
            segment_chunks=config_field(config, "segment_chunks", int, 0),
        )

    @property
    def aux_heads(self) -> list[str]:
        """Heads needing a *per-frame* target, so the dataset caches and serves one.

        This used to exclude both binary heads, on the reasoning that the structured
        NLL supervises them through the segmentation itself. **That was wrong for both,
        and measurably so** -- see the weight constants for the numbers. Onset was the
        worse of the two despite the intuition that ``w_on`` reads the boundary signal
        and boundaries *are* the objective: AP 0.20-0.53 against frame BCE's 0.994.

        The mechanism is that the NLL is discriminative over *segmentations*, so a
        feature earns gradient only in proportion to how much it helps rank gold above
        the alternatives, and ``LinearSpanScorer`` gives every one a learned scalar that
        can shrink toward zero. Nothing asks a head to be *calibrated*, only to be
        useful for ranking -- and a head can be a useful feature channel while being a
        poor detector of the thing it is named for.

        Pitch is the strongest case of all: the span features are invariant to which
        pitch is named, so the NLL *cannot* supervise its values however large the
        weight.

        Set the corresponding weight to 0 to drop a head from this list.
        """
        heads = ["pitch"] if self.pitch else []
        if self.onset_weight:
            heads.append("onset")
        if self.voicing_weight:
            heads.append("voicing")
        return heads

    @property
    def params(self) -> dict[str, object]:
        return {
            "pitch": self.pitch,
            "harmonics": ",".join(str(h) for h in self.harmonics),
            "chunk_frames": CHUNK_FRAMES,
            "pitch_weight": self.pitch_weight,
            "voicing_weight": self.voicing_weight,
            "onset_weight": self.onset_weight,
            # Both, because they are different facts that happen to share a number.
            # `segment_chunks` says the gradient reaches through the carry at all --
            # the difference between descending this objective and ascending it -- while
            # `scorer` says what the potentials are. A run recorded without them is one
            # nobody can later tell apart from the arm it was meant to beat.
            "segment_chunks": self.segment_chunks,
            "scorer": self.decoder.scorer.config.get("kind", "linear"),
        }

    def build_model(self) -> FrameNet:
        # Same encoder; only the objective differs. Assembled by `models`, so this
        # extension never reaches sideways into the frame objective to get it.
        return build_framenet(self.in_channels, self.harmonics, self.pitch)

    def build_loss(self) -> Loss[chunked_ds.ChunkBatch]:
        return SemiCRFLoss(
            self.decoder,
            # Undetached exactly when the chains are bounded. Detaching is what makes
            # the gradient truncated, and truncation is only affordable because the
            # chain is short enough to backprop through -- so the two are one decision.
            detach_carry=self.segment_chunks == 0,
            aux_heads=self.aux_heads,
            # Filtered to the declared heads: a zero weight drops the head from
            # `aux_heads` entirely, and naming a weight for an undeclared head is
            # refused by the loss (a setting that silently does nothing).
            aux_weights={
                h: w for h, w in (
                    ("pitch", self.pitch_weight),
                    ("voicing", self.voicing_weight),
                    ("onset", self.onset_weight),
                ) if h in self.aux_heads
            },
            pos_weight=self.pos_weight,
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
        store: Store,
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

        Nothing below ``train_items`` knows a track's condition -- a ``TrainItem`` is
        feature paths plus notes -- and ``resolve()`` has already rejected a mixed-width
        set, so the windows stack identically. The frame path has always trained on the
        union this way; the structured objective was the odd one out, which made any
        comparison against a frame baseline vary two things at once.
        """
        # Before the corpus sweep, not after: this is a settings contradiction, and
        # finding it takes no data. Checked here rather than in `__init__` because
        # `accumulate_steps` belongs to the loop's config, which the recipe first sees
        # at this call.
        if self.segment_chunks and config.accumulate_steps != self.segment_chunks:
            raise ValueError(
                f"segment_chunks={self.segment_chunks} needs accumulate_steps to match "
                f"it, got {config.accumulate_steps}. The carry is undetached within a "
                "segment, so the optimiser step -- and the single backward it triggers "
                "-- has to land exactly on a segment boundary: earlier and the next "
                "chunk reads a freed graph, later and the step spans a reset."
            )
        dmax, right = self._geometry
        items: list[TrainItem] = []
        for c in conditions:
            items += train_items(
                c.dataset(root, split, limit), c, store, heads=self.aux_heads
            )
        return chunked_ds.ChunkDataset(
            items,
            chunk_frames=CHUNK_FRAMES, dmax=dmax, stride=self.decoder.stride,
            batch_size=config.batch_size, right_context=right,
            heads=self.aux_heads,
            labels=self.decoder.scorer.labels,
            segment_chunks=self.segment_chunks,
        )

    def _loader(self, dataset: chunked_ds.ChunkDataset) -> DataLoader[ChunkBatch]:
        # batch_size=None: the dataset emits whole ChunkBatch objects, because chunks
        # must arrive in per-stream order and a sampler cannot express that. Workers
        # each own a disjoint slice of tracks and stamp globally unique stream ids.
        return DataLoader(
            dataset, batch_size=None, num_workers=0, collate_fn=None,
        )

    def train_loader(
        self, root: str, store: Store, config: TrainConfig, limit: int | None
    ) -> DataLoader[ChunkBatch]:
        return self._loader(
            self._chunks(root, "train", self.conditions, store, config, limit)
        )

    def val_loader(
        self, root: str, store: Store, config: TrainConfig, val_limit: int | None
    ) -> DataLoader[ChunkBatch]:
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        return self._loader(
            self._chunks(root, "validation", [condition], store, config, val_limit)
        )

    def decoder_from(self, state: Mapping[str, Any]) -> Decoder:
        decoder = SemiCRFDecoder(
            stride=self.decoder.stride,
            scorer=build_scorer(self.decoder.scorer.config),
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
