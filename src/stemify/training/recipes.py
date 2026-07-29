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
from pathlib import Path

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from stemify.cache import Cache
from stemify.data.slakh import SlakhDataset
from stemify.decoders.base import Decoder
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.encoders.heads import onset_head, voicing_head
from stemify.encoders.networks import TCN
from stemify.encoders.tagger import FrameTagger
from stemify.training.config import TrainConfig
from stemify.training.datasets import frame as frame_ds
from stemify.training.datasets import sequence as sequence_ds
from stemify.training.examples import prepare_examples
from stemify.training.losses import FrameBCELoss, Loss
from stemify.training.losses.segmental import SemiCRFLoss
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

HEADS = ["onset", "voicing"]

# tag -> (Slakh source, input transform). The target is always the clean stem MIDI,
# regardless of which source the *features* come from -- that asymmetry is the domain
# randomisation. Lives here rather than in train.py so the dependency runs one way
# (train imports recipes, never the reverse).
SOURCES: dict[str, tuple[str, AudioTransform]] = {
    "stem": ("stem", Identity()),
    "mix": ("mix", Identity()),
    "demucs": ("mix", DemucsSeparator()),
}


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
        sources: list[str],
        cache: Cache,
        config: TrainConfig,
        limit: int | None,
    ) -> DataLoader:
        """Batches from the union of the requested source conditions.

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
        backbone = TCN()
        return FrameTagger(
            backbone,
            {
                "onset": onset_head(backbone.hidden),
                "voicing": voicing_head(backbone.hidden),
            },
        )

    def build_loss(self, config: TrainConfig) -> Loss:
        return FrameBCELoss(HEADS, config.pos_weight)

    def _windows(self, paths: list[Path], config: TrainConfig):
        return frame_ds.FrameWindowDataset(
            paths, HEADS, config.window_frames, config.hop_frames
        )

    def train_loader(self, root, sources, cache, config, limit) -> DataLoader:
        paths: list[Path] = []
        for tag in sources:
            source, transform = SOURCES[tag]
            ds = SlakhDataset(root, split="train", source=source, limit=limit)
            paths += prepare_examples(ds, HEADS, cache, transform)
        return DataLoader(
            self._windows(paths, config), shuffle=True,
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        ds = SlakhDataset(root, split="validation", source="stem", limit=val_limit)
        paths = prepare_examples(ds, HEADS, cache)
        return DataLoader(
            self._windows(paths, config),
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

    def __init__(self, decoder: SemiCRFDecoder | None = None) -> None:
        self.decoder = decoder or SemiCRFDecoder()

    def build_model(self) -> nn.Module:
        return FrameRecipe().build_model()  # same encoder; only the objective differs

    def build_loss(self, config: TrainConfig) -> Loss:
        return SemiCRFLoss(self.decoder)

    def _sequences(
        self, root: str, split: str, sources: list[str], cache: Cache, limit: int | None
    ):
        parts = []
        for tag in sources:
            source, transform = SOURCES[tag]
            ds = SlakhDataset(root, split=split, source=source, limit=limit)
            parts.append(sequence_ds.SequenceDataset(ds, cache, transform))
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    def train_loader(self, root, sources, cache, config, limit) -> DataLoader:
        return DataLoader(
            self._sequences(root, "train", sources, cache, limit), shuffle=True,
            collate_fn=sequence_ds.collate, **_loader_kwargs(config),
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        return DataLoader(
            self._sequences(root, "validation", ["stem"], cache, val_limit),
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
