"""``FrameRecipe``: per-frame BCE over fixed windows, grouped by the heuristic.

The paradigm that produced tcn-v1/tcn-v2 and every result in ``docs/experiments.md``
outside the semi-CRF section. Windows are cheap and plentiful, so this is the path
with the gradient steps; the cue->note step is the greedy ``HeuristicDecoder``.
"""

from collections.abc import Sequence
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.encoders.heads import onset_head, voicing_head
from stemify.encoders.networks import TCN
from stemify.encoders.tagger import FrameTagger
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.datasets import frame as frame_ds
from stemify.training.examples import prepare_examples
from stemify.training.losses import FrameBCELoss, Loss
from stemify.training.recipes.base import HEADS, Recipe, _loader_kwargs


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
        # for -- a 2-channel model cannot be fed 1-channel val windows.
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        tracks = self._tracks(root, "validation", cache, condition, val_limit)
        return DataLoader(
            self._windows(tracks, config),
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def decoder_from(self, state: dict) -> Decoder:
        return HeuristicDecoder()


