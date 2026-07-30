"""``Recipe``: the objective and everything that must move with it, as one unit.

Swapping training paradigm is not one choice but four coupled ones -- the batch shape,
the loss, the decoder that scores a checkpoint, and what a checkpoint has to contain.
A flat CLI (``--loss X --dataset Y``) would let you name combinations that cannot
work: ``FrameBCELoss`` over a ``SequenceBatch`` type-checks as strings and fails at
epoch 0, or worse broadcasts into silence. Binding them into a ``Recipe`` makes the
valid combinations the only expressible ones, and reduces the CLI to ``--objective``.

Two exist, and the plan's own advice against speculative generality applies: this is a
seam because there are genuinely two implementations, not because there might be more.

  - ``FrameRecipe``    -- per-frame BCE over fixed windows; the heuristic groups the
    cues into notes afterwards. What produced tcn-v1/tcn-v2.
  - ``SemiCRFRecipe``  -- the structured NLL over whole tracks, training the encoder
    and the segmental decoder jointly under one optimiser.

The ``decoder_from`` member is the one an earlier version of this code got wrong by
omission: the async validator hardcoded the heuristic, so a semi-CRF run would have
reported the *heuristic's* note F1 over its learned cues -- plausible numbers for a
model nobody trained.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.encoders.heads import onset_head, voicing_head
from stemify.encoders.networks import TCN
from stemify.encoders.tagger import FrameTagger
from stemify.training.conditions import CONDITIONS, resolve
from stemify.training.config import TrainConfig
from stemify.training.datasets import frame as frame_ds
from stemify.training.datasets import sequence as sequence_ds
from stemify.training.examples import prepare_examples
from stemify.training.losses import FrameBCELoss, Loss
from stemify.training.losses.segmental import SemiCRFLoss

HEADS = ["onset", "voicing"]


def _loader_kwargs(config: TrainConfig) -> dict:
    """DataLoader settings both paradigms share.

    ``persistent_workers`` keeps the pool alive between epochs -- without it the fork
    cost is paid every epoch and cancels the benefit on short epochs -- and is only
    valid when there are workers, so it is gated."""
    kwargs: dict = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


class Recipe(ABC):
    """One training paradigm: the four choices that have to agree with each other."""

    name: str

    def __init__(self, conditions: Sequence[str] | None = None) -> None:
        """``conditions`` are the input conditions to train over, by tag.

        Held on the recipe rather than passed to ``train_loader`` because the model
        has to be built from them too: a mix-conditioned condition means a wider
        first conv, and a recipe that took the conditions only at loader time would
        already have built a model with the wrong input shape.
        """
        self.conditions = resolve(list(conditions or ["stem", "mix"]))
        # resolve() has already rejected a mixed-width set, so this is unambiguous.
        self.in_channels = self.conditions[0].in_channels

    @abstractmethod
    def build_model(self) -> nn.Module:
        """The encoder. Shared shape across recipes -- what varies is the objective
        over its outputs, not the backbone."""

    @abstractmethod
    def build_loss(self, config: TrainConfig) -> Loss:
        """The objective. May own parameters of its own; ``Trainer`` spans them."""

    @abstractmethod
    def train_loader(
        self,
        root: str,
        cache: Cache,
        config: TrainConfig,
        limit: int | None,
    ) -> DataLoader:
        """Batches from the union of this recipe's input conditions.

        Takes the corpus and the cache rather than pre-built paths, because the two
        paradigms need different things from them: the frame path wants per-frame
        target npz paths, the sequence path wants whole tracks with their contours."""

    @abstractmethod
    def val_loader(
        self, root: str, cache: Cache, config: TrainConfig, val_limit: int | None
    ) -> DataLoader:
        """The inline early-stopping set. Clean stems only -- it is the cheap proxy,
        not the number that matters (see ``Trainer``'s docstring)."""

    @abstractmethod
    def decoder_from(self, state: dict) -> Decoder:
        """The decoder a checkpoint should be *scored* through.

        ``state`` is a ``Trainer.state_cpu`` snapshot, so a structured objective can
        rebuild its trained potentials from ``state["loss"]`` rather than being scored
        through a freshly initialised decoder."""


class FrameRecipe(Recipe):
    """Per-frame BCE over fixed windows; the heuristic groups cues into notes."""

    name = "frame"

    def build_model(self) -> nn.Module:
        backbone = TCN(in_channels=self.in_channels)
        return FrameTagger(
            backbone,
            {
                "onset": onset_head(backbone.hidden),
                "voicing": voicing_head(backbone.hidden),
            },
        )

    def build_loss(self, config: TrainConfig) -> Loss:
        return FrameBCELoss(HEADS, config.pos_weight)

    def _windows(self, tracks: list[tuple[Path, ...]], config: TrainConfig):
        return frame_ds.FrameWindowDataset(
            tracks, HEADS, config.window_frames, config.hop_frames
        )

    def _tracks(
        self, root: str, split: str, cache: Cache, condition, limit: int | None
    ) -> list[tuple[Path, ...]]:
        """One tuple of cached CQT paths per track, ordered by input channel.

        Targets are layered onto channel 0's npz -- it is the audio the labels
        describe -- and the remaining channels contribute features only."""
        ds = condition.dataset(root, split, limit)
        prepare_examples(ds, HEADS, cache, condition.primary.transform)
        return condition.channel_paths(ds, cache)

    def train_loader(self, root, cache, config, limit) -> DataLoader:
        tracks: list[tuple[Path, ...]] = []
        for condition in self.conditions:
            tracks += self._tracks(root, "train", cache, condition, limit)
        return DataLoader(
            self._windows(tracks, config), shuffle=True,
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        # Clean stems, but through the *same channel layout* the model was built
        # for -- a 2-channel model cannot be fed 1-channel val windows. "stem" and
        # "stem+mix" are the respective clean-bass conditions.
        condition = CONDITIONS["stem+mix" if self.in_channels == 2 else "stem"]
        tracks = self._tracks(root, "validation", cache, condition, val_limit)
        return DataLoader(
            self._windows(tracks, config),
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def decoder_from(self, state: dict) -> Decoder:
        return HeuristicDecoder()


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
        # The sequence dataset feeds the encoder one CQT per track, so a second
        # input channel has nowhere to go on this path yet. Reject it here rather
        # than let a 2-channel model meet 1-channel batches at the first conv.
        if self.in_channels != 1:
            raise ValueError(
                "the semicrf objective has no multi-channel sequence loader yet; "
                f"conditions {[c.name for c in self.conditions]} need "
                f"{self.in_channels} channels"
            )
        self.decoder = decoder or SemiCRFDecoder()

    def build_model(self) -> nn.Module:
        # Same encoder; only the objective differs.
        return FrameRecipe([c.name for c in self.conditions]).build_model()

    def build_loss(self, config: TrainConfig) -> Loss:
        return SemiCRFLoss(self.decoder)

    def _sequences(self, root: str, split: str, conditions, cache, limit: int | None):
        parts = [
            sequence_ds.SequenceDataset(
                c.dataset(root, split, limit), cache, c.primary.transform
            )
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
            self._sequences(root, "validation", [CONDITIONS["stem"]], cache, val_limit),
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


RECIPES: dict[str, type[Recipe]] = {
    FrameRecipe.name: FrameRecipe,
    SemiCRFRecipe.name: SemiCRFRecipe,
}
