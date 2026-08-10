"""``FrameRecipe``: per-frame BCE over fixed windows, grouped by the greedy decoder.

The paradigm that produced tcn-v1/tcn-v2 and every result in ``docs/experiments.md``
outside the semi-CRF section. Windows are cheap and plentiful, so this is the path
with the gradient steps; the cue->note step is the greedy ``GreedyDecoder``.
"""

from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from torch.utils.data import DataLoader

from stemify.persistence.store import Store
from stemify.transcription.decoders.base import Decoder
from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.models.assembly import build_cuenet
from stemify.models.cuenet import CueNet
from stemify.fitting.base import (
    Loss,
    Recipe,
    config_field,
    config_int_list,
    config_str_list,
)
from stemify.core.conditions import CLEAN_BY_WIDTH, CONDITIONS, Condition
from stemify.fitting.config import TrainConfig
from stemify.core.examples import prepare_examples
from stemify.training.frame import dataset as frame_ds
from stemify.training.frame.dataset import FrameBatch
from stemify.training.frame.loss import FrameBCELoss


# A window is one training example: enough frames for context, short enough to batch.
# 256 frames ~= 3 s at the 11.6 ms hop, and a 50% hop puts an onset that sat near one
# window's edge in the middle of the next.
WINDOW_FRAMES = 256
HOP_FRAMES = 128


class FrameRecipe(Recipe):
    """Per-frame BCE over fixed windows; ``GreedyDecoder`` groups cues into notes."""

    name = "frame"

    def __init__(
        self,
        conditions: Sequence[str] | None = None,
        pitch: bool = False,
        window_frames: int = WINDOW_FRAMES,
        hop_frames: int = HOP_FRAMES,
        pos_weight: dict[str, float | None] | None = None,
        harmonics: Sequence[int] | None = None,
        stem_dropout: float = 0.0,
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
        # `(1,)` is the plain CQT: one channel per rendering, full frequency axis, and
        # `_ConvFrontend.stack` returns its input untouched. Kept as the default so
        # every checkpoint trained before harmonic stacking rebuilds unchanged.
        self.harmonics = tuple(harmonics or (1,))
        # Per-window probability of silencing each context channel, so a bichannel
        # model can also transcribe an isolated bass. Recorded in `config` because it
        # changes the input distribution: a checkpoint has to say whether it was
        # trained under it, or two arms differing only in this look identical.
        self.stem_dropout = float(stem_dropout)

    @property
    def heads(self) -> list[str]:
        """The binary pair, plus pitch when this run trains it.

        The single list the cached targets, the loss terms and the checkpoint's
        ``heads`` field are all derived from, so they cannot drift apart.
        """
        return [*super().heads, "pitch"] if self.pitch else super().heads

    @property
    def config(self) -> dict[str, object]:
        """Everything ``__init__`` takes, unflattened -- see ``Recipe.config``."""
        return {
            **super().config,
            "pitch": self.pitch,
            "window_frames": self.window_frames,
            "hop_frames": self.hop_frames,
            "pos_weight": dict(self.pos_weight),
            "harmonics": list(self.harmonics),
            "stem_dropout": self.stem_dropout,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "FrameRecipe":
        """Field by field -- see ``Recipe.from_config`` for why not ``**config``."""
        weights = config.get("pos_weight") or {}
        if not isinstance(weights, Mapping):
            raise ValueError(
                "recipe config field 'pos_weight' should be a mapping of "
                f"head -> weight, got {type(weights).__name__}"
            )
        return cls(
            conditions=config_str_list(config, "conditions"),
            pitch=config_field(config, "pitch", bool, False),
            window_frames=config_field(config, "window_frames", int, WINDOW_FRAMES),
            hop_frames=config_field(config, "hop_frames", int, HOP_FRAMES),
            pos_weight={str(h): float(w) for h, w in weights.items()},
            harmonics=[int(h) for h in config_int_list(config, "harmonics")] or None,
            stem_dropout=config_field(config, "stem_dropout", float, 0.0),
        )

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
            "harmonics": ",".join(str(h) for h in self.harmonics),
        }

    def build_model(self) -> CueNet:
        return build_cuenet(self.in_channels, self.harmonics, self.pitch)

    def build_loss(self) -> Loss[FrameBatch]:
        return FrameBCELoss(self.heads, self.pos_weight)

    def _windows(
        self, tracks: list[tuple[Path, ...]], train: bool
    ) -> frame_ds.FrameWindowDataset:
        """``train`` gates stem dropout, which is an augmentation and belongs to the
        training split alone. Applying it to validation would move the metric under the
        model rather than measuring it -- val F1 would drift with the dropout rate and
        no longer compare against any other arm."""
        return frame_ds.FrameWindowDataset(
            tracks, self.heads, self.window_frames, self.hop_frames,
            stem_dropout=self.stem_dropout if train else 0.0,
        )

    def _tracks(
        self,
        root: str,
        split: str,
        store: Store,
        condition: Condition,
        limit: int | None,
    ) -> list[tuple[Path, ...]]:
        """One tuple of cached CQT paths per track, ordered by input channel.

        Targets are layered onto channel 0's npz -- it is the audio the labels
        describe -- and the remaining channels contribute features only."""
        ds = condition.dataset(root, split, limit)
        prepare_examples(ds, self.heads, store, condition.primary.transform)
        return condition.channel_paths(ds, store)

    def _loader(
        self, tracks: list[tuple[Path, ...]], config: TrainConfig, shuffle: bool
    ) -> DataLoader[FrameBatch]:
        """This strategy's loader, with the settings spelled out as arguments.

        These used to come from a shared ``_loader_kwargs(config)`` in ``fitting.base``
        splatted in as ``**dict[str, object]``, on the reasoning that both paradigms
        needed the same DataLoader settings. They do not: the semi-CRF's loader is
        ``batch_size=None, num_workers=0`` because its dataset emits whole batches in
        stream order, so it never called this and the sharing was notional. A splatted
        ``dict[str, object]`` also erases every argument's type at the call, which is
        how a setting can be renamed here and silently ignored by ``DataLoader``.

        ``persistent_workers`` is only valid with workers, and without it the fork cost
        is paid every epoch -- which cancels the benefit on short epochs.
        """
        return DataLoader(
            self._windows(tracks, train=shuffle),
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            collate_fn=frame_ds.collate,
        )

    def train_loader(
        self, root: str, store: Store, config: TrainConfig, limit: int | None
    ) -> DataLoader[FrameBatch]:
        tracks: list[tuple[Path, ...]] = []
        for condition in self.conditions:
            tracks += self._tracks(root, "train", store, condition, limit)
        return self._loader(tracks, config, shuffle=True)

    def val_loader(
        self, root: str, store: Store, config: TrainConfig, val_limit: int | None
    ) -> DataLoader[FrameBatch]:
        # Clean stems, but through the *same channel layout* the model was built
        # for -- a 2-channel model cannot be fed 1-channel val windows.
        condition = CONDITIONS[CLEAN_BY_WIDTH[self.in_channels]]
        tracks = self._tracks(root, "validation", store, condition, val_limit)
        return self._loader(tracks, config, shuffle=False)

    def decoder_from(self, state: Mapping[str, Any]) -> Decoder:
        return GreedyDecoder()
