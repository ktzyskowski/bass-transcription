"""What a training *strategy* has to implement: ``Recipe`` and ``Loss``.

The two contracts live together because they are two halves of one thing -- a
``Recipe`` *builds* a ``Loss`` (``build_loss``), and a strategy that changes one almost
always changes the other. Splitting them across packages is what made the semi-CRF
spread over three directories before the training tree was sliced by strategy.

The implementations live in ``training/frame`` and ``training/semicrf``, one package
per paradigm; ``training/recipes.py`` is the table that names them. This module knows
about neither, which is what lets a strategy package import it without a cycle.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.training.conditions import resolve
from stemify.training.config import TrainConfig

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


class Loss(ABC):
    """A training objective the ``Trainer`` optimises, agnostic to batch shape."""

    @abstractmethod
    def compute(
        self, model: nn.Module, batch: object, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``(loss, metrics)``: the scalar to backward, and per-component scalars to
        log. ``metrics`` must include a ``"total"`` key -- the number the early-stopper
        monitors -- so the loop never reaches into a paradigm-specific breakdown."""

    @abstractmethod
    def batch_size(self, batch: object) -> int:
        """How many examples ``batch`` holds, for the example-weighted epoch mean."""

    def parameters(self) -> Iterable[nn.Parameter]:
        """Learnable parameters this objective adds beyond the model (none by
        default; a structured decoder overrides to return its own)."""
        return ()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The objective's own learned state, for snapshotting and persistence.

        The companion to ``parameters``: anything the optimiser is allowed to move
        must also be capturable, or it is trained and then lost. A structured
        decoder's potentials are the model in every sense that matters at decode
        time -- an encoder saved without them pairs with a *freshly initialised*
        decoder, which still emits notes and so fails silently rather than loudly."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore what ``state_dict`` captured. No-op for a parameterless objective."""
        if state:
            raise ValueError(
                f"{type(self).__name__} holds no parameters but was given state "
                f"{sorted(state)}"
            )

    def to(self, device: torch.device) -> "Loss":
        """Move any owned modules onto ``device`` (no-op unless the objective holds
        parameters). Returns self for chaining, like ``nn.Module.to``."""
        return self


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
        # ``clean,separated`` rather than the old ``stem,mix``: those were retired tags
        # resolving to ``clean-multibass`` (kept only to reproduce tcn-v1/v2, and
        # comparable to nothing else) and ``mixture`` (the un-separated negative
        # control, which scores 0.27) -- so the default trained on the two conditions
        # least worth training on, with a deprecation warning for each.
        self.conditions = resolve(list(conditions or ["clean", "separated"]))
        # resolve() has already rejected a mixed-width set, so this is unambiguous.
        self.in_channels = self.conditions[0].in_channels

    @property
    def heads(self) -> list[str]:
        """The heads this run trains. The binary pair unless a strategy says otherwise.

        A property rather than a constant because a strategy that can train more (the
        frame path, with ``--pitch``) has to drive three things from one list: the
        target cached per example, the loss terms built, and the names recorded in the
        checkpoint. A checkpoint whose ``heads`` disagrees with the branches it holds
        loads *silently* without them.

        Deliberately does not mention pitch here. Only the frame strategy has that
        head -- the semi-CRF's DP scores onset and voicing evidence and would never
        read it -- so naming it in the base would be the contract advertising a
        capability most of its implementations do not have.
        """
        return list(HEADS)

    @abstractmethod
    def build_model(self) -> nn.Module:
        """The encoder. Shared shape across recipes -- what varies is the objective
        over its outputs, not the backbone."""

    @abstractmethod
    def build_loss(self) -> Loss:
        """The objective. May own parameters of its own; ``Trainer`` spans them.

        Takes nothing: a strategy's own knobs live on the recipe (see
        ``FrameRecipe``), so the only thing this needs is already ``self``.
        """

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
