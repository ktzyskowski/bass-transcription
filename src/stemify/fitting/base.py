"""What a training *strategy* has to implement: ``Recipe`` and ``Loss``.

The two contracts live together because they are two halves of one thing -- a
``Recipe`` *builds* a ``Loss`` (``build_loss``), and a strategy that changes one almost
always changes the other. Splitting them across packages is what made the semi-CRF
spread over three directories before the training tree was sliced by strategy.

The implementations live in ``training/frame`` and ``stemify/semicrf/fit``, one package
per paradigm; ``training/recipes.py`` is the table that names them. This module knows
about neither, which is what lets a strategy package import it without a cycle.

That the semi-CRF's half sits outside this package is the point of the split: it is an
extension, so it depends on these contracts rather than living beside them. These three
names -- ``Recipe``, ``Loss``, ``Batch`` -- plus ``TrainConfig`` are the entire surface
an objective needs, and they are deliberately the only edges from ``semicrf`` into
``training``.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader

from stemify.persistence.store import Store
from stemify.transcription.decoders.base import Decoder
from stemify.models.framenet import FrameNet
from stemify.core.conditions import resolve
from stemify.fitting.config import TrainConfig

HEADS = ["onset", "voicing"]

GRAD_NORM = "grad_norm"
"""The one batch-metric key contributed by the training loop rather than a ``Loss``.

Lives beside the ``compute`` contract because it is part of it: everything else in that
mapping is a term of the objective, and a consumer treating the whole mapping as a
per-head loss breakdown (the console log, the MLflow ``loss/`` namespace) needs to know
which key is not. ``Trainer`` fills it with the pre-clip gradient norm.
"""

T = TypeVar("T")


def config_field(
    config: Mapping[str, object], key: str, kind: type[T], default: T
) -> T:
    """One field out of a ``Recipe.config`` mapping, checked.

    The whole of the coercion helper set on purpose. A checkpoint's config is plain
    data read off disk, so a wrong type there is a real possibility rather than a
    theoretical one -- but it is also *rare*, so the answer is one function that names
    the offending field, not a schema library.

    ``bool`` is checked before ``int`` by Python's own subclassing rule (``bool`` is an
    ``int``), which is why the check is ``isinstance`` against the requested type
    rather than a coercing ``kind(value)``: ``int(True)`` would silently accept a bool
    where a count was meant.
    """
    value = config.get(key, default)
    if not isinstance(value, kind):
        raise ValueError(
            f"recipe config field {key!r} should be {kind.__name__}, got "
            f"{type(value).__name__} ({value!r})"
        )
    return value


def config_str_list(
    config: Mapping[str, object], key: str, default: Sequence[str] | None = None
) -> list[str]:
    """A list-of-strings field. Separate from ``config_field`` because a container
    needs its *elements* checked too, and ``isinstance(x, list)`` says nothing about
    them."""
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"recipe config is missing required field {key!r}")
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(
            f"recipe config field {key!r} should be a list of str, got {value!r}"
        )
    items = list(value)
    bad = [v for v in items if not isinstance(v, str)]
    if bad:
        raise ValueError(f"recipe config field {key!r} has non-str entries: {bad!r}")
    return [str(v) for v in items]


def config_int_list(
    config: Mapping[str, object], key: str, default: Sequence[int] = ()
) -> list[int]:
    """A list-of-ints field -- the peer of ``config_str_list``.

    Absent means *empty* rather than an error: an int list here names an optional
    setting (``harmonics``), so a checkpoint written before that setting existed must
    still load and fall back to the recipe's own default. ``bool`` is rejected
    explicitly, since it is an ``int`` to ``isinstance`` and would quietly become 0/1.
    """
    value = config.get(key, default)
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(
            f"recipe config field {key!r} should be a list of int, got {value!r}"
        )
    items = list(value)
    bad = [v for v in items if isinstance(v, bool) or not isinstance(v, int)]
    if bad:
        raise ValueError(f"recipe config field {key!r} has non-int entries: {bad!r}")
    return [int(v) for v in items]


def epoch_batches(loader: DataLoader[Any]) -> int | str:
    """Batches per epoch, or ``"streaming"`` when that is not knowable in advance.

    The chunked semi-CRF loader wraps an ``IterableDataset``: it walks ``batch_size``
    parallel streams and stops when the first one runs out, so its epoch length is a
    property of how the tracks divided up rather than something ``len`` can answer.
    Logged as a param either way -- it is the number that makes "10 epochs" comparable
    between objectives, and assuming every loader is sized is what crashed a run at
    the first MLflow write while every smoke test (which ran without a tracking URI)
    passed."""
    try:
        return len(loader)
    except TypeError:
        return "streaming"


class Batch(Protocol):
    """What the training loop needs of a batch, which is one thing: its size.

    Everything else a batch holds is private between the loader that built it and the
    loss that reads it -- ``Trainer`` passes it through unopened. So this is the whole
    contract, and each strategy's batch type is free to look however its objective
    needs (``FrameBatch`` is windows and per-frame targets; ``ChunkBatch`` carries an
    alpha carry's worth of stream bookkeeping besides).

    ``__len__`` rather than a ``size()`` method because a batch *is* a collection of
    examples, and because this used to be ``Loss.batch_size(batch)`` -- a property of
    the batch, implemented on the loss. For the chunked objective that method had
    become pure forwarding to ``len``; for the frame path it existed only because the
    batch was an anonymous tuple, where ``len`` would have answered 2.
    """

    def __len__(self) -> int:
        """Examples in this batch, for the example-weighted epoch mean."""
        ...


B = TypeVar("B", bound=Batch)


class Loss(ABC, Generic[B]):
    """A training objective the ``Trainer`` optimises, over one strategy's batch type.

    Generic in that type so an objective states what it consumes: ``Loss[FrameBatch]``
    reads windows, ``Loss[ChunkBatch]`` reads ordered chunks, and neither takes
    ``object``. The pairing is enforced by ``Recipe``, which builds the loader and the
    loss together for exactly this reason.
    """

    @abstractmethod
    def compute(
        self, model: nn.Module, batch: B, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``(loss, metrics)``: the scalar to backward, and per-component scalars to
        log. ``metrics`` must include a ``"total"`` key -- the number the early-stopper
        monitors -- so the loop never reaches into a paradigm-specific breakdown."""

    @property
    def couples_micro_batches(self) -> bool:
        """Do consecutive ``compute`` calls share one autograd graph?

        ``False`` -- the default and the stateless case -- lets the loop backward each
        micro-batch and free its graph immediately, which is what bounds activation
        memory to one batch.

        ``True`` says the objective carries a *differentiable* state forward, so
        backwarding a micro-batch would free a graph the next one still needs (torch
        raises "Trying to backward through the graph a second time"). The loop then
        accumulates the loss tensors and backwards once per window instead. That costs
        the window's activations in memory, and it is the only way the gradient reaches
        back through the carried state -- without it a semi-CRF's step is not a
        derivative of the loss it reports.

        A property rather than a constructor flag because it is a fact about the
        objective's graph, not a setting: an objective that answered wrongly would
        either crash immediately (True needed, False given) or silently hold N graphs
        for nothing.
        """
        return False

    def parameters(self) -> Iterable[nn.Parameter]:
        """Learnable parameters this objective adds beyond the model (none by
        default; a structured decoder overrides to return its own)."""
        return ()

    def scalars(self) -> Mapping[str, float]:
        """Interpretable parameters of this objective, to watch per epoch (none by
        default).

        The companion to ``Recipe.params``, one level down: that reports the knobs a
        run was *given*, this reports the few the run *learned* whose values mean
        something on their own. A structured decoder is the case that needs it -- its
        potentials decide the shape of the decode, so a decode that changes character
        between epochs is explained by numbers that otherwise appear only inside a
        checkpoint, and only for the epochs that happened to be best.

        Deliberately not derived from ``state_dict``: that is everything the optimiser
        may move, including whatever a future scorer adds, and a consumer cannot tell
        which entries are worth an axis. Naming them here keeps that judgement with the
        objective that knows.
        """
        return {}

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

    def to(self, device: torch.device) -> "Loss[B]":
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

    @property
    def config(self) -> dict[str, object]:
        """The keyword arguments that reconstruct this recipe.

        The companion to ``from_config``, and the whole of what a checkpoint stores
        about *how* its model was built: ``heads``, ``in_channels`` and the decoder a
        checkpoint is scored through are all derived from a recipe, so a checkpoint
        that records this needs to record nothing else. That is what stops it
        accumulating a field per question anyone ever asks of it.

        Deliberately distinct from ``params``. That one is a *display* projection for
        MLflow -- it renders ``pos_weight`` as ``"onset=5"`` and omits the conditions --
        so it is lossy by design. The two should eventually be one, with ``params``
        derived from this; until then, this is the one that must round-trip.
        """
        return {"conditions": [c.name for c in self.conditions]}

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, object]) -> "Recipe":
        """Rebuild a recipe from ``config``, pulling each field explicitly.

        Deliberately *not* ``cls(**config)``. This is a deserialisation boundary: the
        mapping comes off disk, possibly written by an older version -- ``Checkpoint``
        already reconstructs one for pre-v2 files -- so nothing guarantees a key holds
        the type the constructor declares. Splatting an untyped mapping into a typed
        signature is unsound, and a type checker says so. Pulling each field coerces it
        where it is read, and a malformed checkpoint fails here, naming the field,
        instead of somewhere downstream with a shape error.

        Abstract rather than defaulted: a recipe that cannot be rebuilt from its own
        ``config`` is broken, and that should be a missing method rather than a
        checkpoint that loads into the wrong model.
        """

    @property
    def params(self) -> dict[str, object]:
        """Strategy-specific settings worth recording on a run.

        The entry point merges these into the run's logged params because it cannot
        know them: window length means nothing to the semi-CRF, chunk length means
        nothing to the frame path, and a run tagged with a knob its strategy never read
        is worse than one tagged with neither. A strategy that gains a knob records it
        here, so the thing that owns the value is also the thing that reports it.
        """
        return {}

    @abstractmethod
    def build_model(self) -> FrameNet:
        """The encoder. Shared shape across recipes -- what varies is the objective
        over its outputs, not the backbone.

        ``FrameNet`` rather than ``nn.Module`` because that is what the sentence
        above already claims, and what every consumer needs: ``LearnedEncoder`` takes a
        ``FrameNet``, so a weaker return type here bought no freedom and instead
        put an unchecked assumption at each of its two call sites."""

    @abstractmethod
    def build_loss(self) -> "Loss[Any]":
        """The objective. May own parameters of its own; ``Trainer`` spans them.

        Takes nothing: a strategy's own knobs live on the recipe (see
        ``FrameRecipe``), so the only thing this needs is already ``self``.
        """

    @abstractmethod
    def train_loader(
        self,
        root: str,
        store: Store,
        config: TrainConfig,
        limit: int | None,
    ) -> "DataLoader[Any]":
        """Batches from the union of this recipe's input conditions.

        Takes the corpus and the store rather than pre-built paths, because the two
        paradigms need different things from them: the frame path wants per-frame
        target npz paths, the sequence path wants whole tracks with their contours."""

    @abstractmethod
    def val_loader(
        self, root: str, store: Store, config: TrainConfig, val_limit: int | None
    ) -> "DataLoader[Any]":
        """The inline early-stopping set. Clean stems only -- it is the cheap proxy,
        not the number that matters (see ``Trainer``'s docstring)."""

    @abstractmethod
    def decoder_from(self, state: Mapping[str, Any]) -> Decoder:
        """The decoder a checkpoint should be *scored* through.

        ``state`` is a ``Trainer.state_cpu`` snapshot, so a structured objective can
        rebuild its trained potentials from ``state["loss"]`` rather than being scored
        through a freshly initialised decoder."""
