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
from stemify.models.heads import onset_head, pitch_head, voicing_head
from stemify.models.networks import TCN
from stemify.models.tagger import FrameTagger
from stemify.training.base import Loss, Recipe, _loader_kwargs
from stemify.training.conditions import CLEAN_BY_WIDTH, CONDITIONS
from stemify.training.config import TrainConfig
from stemify.training.examples import prepare_examples
from stemify.training.frame import dataset as frame_ds
from stemify.training.frame.loss import FrameBCELoss


# A window is one training example: enough frames for context, short enough to batch.
# 256 frames ~= 3 s at the 11.6 ms hop, and a 50% hop puts an onset that sat near one
# window's edge in the middle of the next.
WINDOW_FRAMES = 256
HOP_FRAMES = 128


class FrameRecipe(Recipe):
    """Per-frame BCE over fixed windows; the heuristic groups cues into notes."""

    name = "frame"

    def __init__(
        self,
        conditions: Sequence[str] | None = None,
        pitch: bool = False,
        window_frames: int = WINDOW_FRAMES,
        hop_frames: int = HOP_FRAMES,
        pos_weight: dict[str, float | None] | None = None,
    ) -> None:
        """These three used to live on ``TrainConfig``, and only this recipe read them.

        The semi-CRF sets its chunk length from its own ``CHUNK_FRAMES`` and its
        objective takes no class weights, so while they were shared settings
        ``--pos-weight`` on a semi-CRF run was accepted and quietly did nothing. Held
        here, handing one to another strategy is a ``TypeError``.

        ``pitch`` is here for the same reason, having previously sat on the ``Recipe``
        base: only this strategy has that head. It changes the model's shape, what
        every cached example must carry (a per-frame MIDI target), and what the
        checkpoint records -- see ``heads``.
        """
        super().__init__(conditions)
        self.pitch = pitch
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self.pos_weight = dict(pos_weight or {})

    @property
    def heads(self) -> list[str]:
        """The binary pair, plus pitch when this run trains it.

        The single list the cached targets, the loss terms and the checkpoint's
        ``heads`` field are all derived from, so they cannot drift apart.
        """
        return [*super().heads, "pitch"] if self.pitch else super().heads

    @property
    def params(self) -> dict[str, object]:
        """What this strategy contributes to the run's logged params."""
        return {
            "window": self.window_frames,
            "hop": self.hop_frames,
            "pitch": self.pitch,
            # Logged so a run records whether it was a candidate generator or a
            # balanced model -- the two are not comparable on F1.
            "pos_weight": ",".join(
                f"{h}={w}" for h, w in sorted(self.pos_weight.items())
            ) or "none",
        }

    def build_model(self) -> nn.Module:
        backbone = TCN(in_channels=self.in_channels)
        return FrameTagger(
            backbone,
            {
                "onset": onset_head(backbone.hidden),
                "voicing": voicing_head(backbone.hidden),
            },
            # A FrequencyHead takes no `hidden`: it taps the front-end before the
            # flatten, reading (B, C, F, T) with the frequency axis still intact.
            frequency_heads={"pitch": pitch_head()} if self.pitch else None,
        )

    def build_loss(self) -> Loss:
        return FrameBCELoss(self.heads, self.pos_weight)

    def _windows(self, tracks: list[tuple[Path, ...]]) -> frame_ds.FrameWindowDataset:
        return frame_ds.FrameWindowDataset(
            tracks, self.heads, self.window_frames, self.hop_frames
        )

    def _tracks(
        self, root: str, split: str, cache: Cache, condition, limit: int | None
    ) -> list[tuple[Path, ...]]:
        """One tuple of cached CQT paths per track, ordered by input channel.

        Targets are layered onto channel 0's npz -- it is the audio the labels
        describe -- and the remaining channels contribute features only."""
        ds = condition.dataset(root, split, limit)
        prepare_examples(ds, self.heads, cache, condition.primary.transform)
        return condition.channel_paths(ds, cache)

    def train_loader(self, root, cache, config, limit) -> DataLoader:
        tracks: list[tuple[Path, ...]] = []
        for condition in self.conditions:
            tracks += self._tracks(root, "train", cache, condition, limit)
        return DataLoader(
            self._windows(tracks), shuffle=True,
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def val_loader(self, root, cache, config, val_limit) -> DataLoader:
        # Clean stems, but through the *same channel layout* the model was built
        # for -- a 2-channel model cannot be fed 1-channel val windows.
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        tracks = self._tracks(root, "validation", cache, condition, val_limit)
        return DataLoader(
            self._windows(tracks),
            collate_fn=frame_ds.collate, **_loader_kwargs(config),
        )

    def decoder_from(self, state: dict) -> Decoder:
        return HeuristicDecoder()
