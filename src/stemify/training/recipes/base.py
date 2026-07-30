"""``Recipe``: the objective and everything that must move with it, as one unit.

Swapping training paradigm is not one choice but four coupled ones -- the batch shape,
the loss, the decoder that scores a checkpoint, and what a checkpoint has to contain.
A flat CLI (``--loss X --dataset Y``) would let you name combinations that cannot
work: ``FrameBCELoss`` over a ``SequenceBatch`` type-checks as strings and fails at
epoch 0, or worse broadcasts into silence. Binding them into a ``Recipe`` makes the
valid combinations the only expressible ones, and reduces the CLI to ``--objective``.

This module holds only the seam and what every recipe shares; the concrete recipes
live one file each beside it (``frame.py``, ``semicrf.py``) and are assembled into the
``RECIPES`` table by the package ``__init__``. That split is why adding a paradigm
does not grow a file every other paradigm has to be read around.

The ``decoder_from`` member is the one an earlier version of this code got wrong by
omission: the async validator hardcoded the heuristic, so a semi-CRF run would have
reported the *heuristic's* note F1 over its learned cues -- plausible numbers for a
model nobody trained.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import nn
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.decoders.base import Decoder
from stemify.training.conditions import resolve
from stemify.training.config import TrainConfig
from stemify.training.losses import Loss

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


