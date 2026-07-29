"""The ``Loss`` interface: a batch -> a scalar to minimise, plus what to log.

The ``Trainer`` is written against this, not against any concrete objective, so a
new training paradigm is a new ``Loss`` (and a matching dataset/collate), never a
new loop. A ``Loss`` owns three things the loop must stay ignorant of:

  - ``compute`` -- the forward from a batch to the scalar loss (and the per-component
    metrics worth charting), including any decoder that lives inside the objective.
  - ``parameters`` -- learnable parameters *beyond* the model, so the optimiser can
    span them (a structured decoder trains jointly with the encoder).
  - ``batch_size`` -- how many examples a batch holds, since only the ``Loss`` knows
    the batch's shape (a window tuple vs a padded sequence batch), and the epoch mean
    weights by it.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn


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
