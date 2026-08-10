"""``Checkpoint``: a trained system on disk, and the one way to reconstitute it.

**Why this type exists.** The same content had two names -- ``Trainer.state_cpu``
returned ``{"model", "loss"}`` while the saved file used ``{"state_dict",
"loss_state"}`` -- and every reader translated between them by hand. Training-time
validation used one shape and the eval CLI used the other, so the CLI hand-rolled its
restore and hand-rolled it into dropping the semi-CRF's potentials entirely: a
default-constructed decoder has ``dur=zeros``, no duration preference at all, and it
still emits notes. Numbers came out; a model nobody trained produced them.

**Why it holds three things and will keep holding three things.** A checkpoint's job
is to answer *"give me the transcriber this run produced"*. Everything that used to be
proposed as a field -- ``heads``, ``in_channels``, ``conditions``, the decoder's stride
and dmax -- is a fact about the ``Recipe`` that built the model. So the file records the
recipe and the two state dicts, and the recipe answers the rest. Anything describing
*how* a model was built belongs in ``Recipe.config``, never as a new field here; that is
what keeps this from growing a column per question.

It lives beside ``Recipe`` rather than at the package root because it is defined by one:
change what a recipe takes and this follows, which is the test for what belongs together.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from stemify.platform.atomic import atomic_write
from stemify.transcription.decoders.base import Decoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.models.cuenet import CueNet
from stemify.transcription.transcriber import Transcriber
from stemify.training.base import Recipe
from stemify.training.recipes import RECIPES

logger = logging.getLogger(__name__)

# Bumped when the on-disk shape changes in a way a reader must know about. Version 1
# is every checkpoint written before this type existed: it named the two state dicts
# differently and recorded derived facts (heads, in_channels) instead of the recipe.
VERSION = 2


@dataclass(frozen=True)
class Checkpoint:
    """A trained system: the recipe that built it, and what training moved."""

    recipe: Recipe
    model_state: dict[str, torch.Tensor]
    # The objective's own parameters -- empty for frame BCE, the potentials for a
    # semi-CRF. Saved because the optimiser trained them alongside the encoder, so a
    # file without them holds half a system while looking complete.
    objective_state: dict[str, torch.Tensor]

    def save(self, path: Path) -> None:
        """Write atomically: a torn checkpoint that still loads is worse than none.

        The two state dicts are both ``dict[str, Tensor]``, so nothing about their
        *types* stops a caller passing them the wrong way round -- and a swapped pair
        writes a file that looks fine until something tries to load it, which is after
        the training run has ended. Checking against the recipe's own model here moves
        that failure to the moment the mistake is made.
        """
        expected = set(self.recipe.build_model().state_dict())
        if set(self.model_state) != expected:
            missing = sorted(expected - set(self.model_state))[:3]
            extra = sorted(set(self.model_state) - expected)[:3]
            raise ValueError(
                f"model_state does not match what {self.recipe.name!r} builds "
                f"(missing {missing}, unexpected {extra}). The usual cause is the two "
                "state dicts passed in the wrong order -- both are dict[str, Tensor], "
                "so nothing else would have caught it."
            )
        payload = {
            "version": VERSION,
            "recipe": {"name": self.recipe.name, "config": self.recipe.config},
            "model_state": self.model_state,
            "objective_state": self.objective_state,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda p: torch.save(payload, p))

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("version") == VERSION:
            spec = raw["recipe"]
            recipe = RECIPES[spec["name"]].from_config(spec["config"])
            return cls(recipe, raw["model_state"], raw["objective_state"])
        return cls._from_v1(raw, path)

    @classmethod
    def _from_v1(cls, raw: dict[str, Any], path: Path) -> "Checkpoint":
        """Read a pre-``Checkpoint`` file, inferring the recipe it never recorded.

        Every inference is logged. A v1 file says which heads it has but not which
        objective trained it, so the presence of objective parameters is the only
        available signal -- sound today, because frame BCE is the one parameterless
        objective. The decoder's geometry was never written at all and falls back to
        today's defaults, which is correct only because nothing has varied them; a run
        that did would be silently mis-scored, and that is exactly why ``VERSION`` 2
        records it.
        """
        objective_state = raw.get("loss_state") or {}
        name = "semicrf" if objective_state else "frame"
        config: dict[str, object] = {
            "conditions": list(raw.get("conditions", ["clean"])),
            "pitch": "pitch" in raw.get("heads", []),
        }
        logger.warning(
            "%s is a version-1 checkpoint: inferring objective=%r, conditions=%s, "
            "pitch=%s, and default decoder geometry. Re-save it to pin these.",
            path, name, config["conditions"], config["pitch"],
        )
        return cls(
            RECIPES[name].from_config(config), raw["state_dict"], objective_state
        )

    # --- reconstitution: the one question every caller asks -----------------------

    def model(self) -> CueNet:
        """The trained encoder network. Strict: a checkpoint whose weights include a
        branch the rebuild left out would otherwise load silently and go on producing
        onset and voicing while quietly having no pitch head."""
        model = self.recipe.build_model()
        model.load_state_dict(self.model_state)
        return model

    def decoder(self) -> Decoder:
        """The decoder this checkpoint should be *scored* through, with its trained
        parameters. The recipe owns the choice, so no caller has to guess it from a
        flag -- which is how a semi-CRF came to be scored through a fresh one."""
        return self.recipe.decoder_from({"loss": self.objective_state})

    def encoder(self, **kwargs: Any) -> LearnedEncoder:
        return LearnedEncoder(self.model(), **kwargs)

    def transcriber(self, **kwargs: Any) -> Transcriber:
        """Audio -> notes, exactly as this run would deploy.

        The seam that training-time validation and the eval CLI must share: two
        implementations of this is what let them disagree.
        """
        return Transcriber(self.encoder(**kwargs), self.decoder())


def load_cuenet(checkpoint: Path) -> CueNet:
    """The trained ``CueNet`` a checkpoint file holds.

    Was ``evaluation.cli.load_tagger``: a library function living in a CLI module,
    which six scripts imported -- so reaching for a checkpoint loader meant importing
    argument parsing. It sits here instead, next to the ``Checkpoint`` it delegates to.

    Renamed with the class. It stopped returning a ``FrameTagger`` when that became
    ``CueNet``, and it stopped rebuilding anything itself before that: it used to
    re-derive the architecture from ``heads`` and ``in_channels``, a second
    implementation of what the training recipe already knows. ``Checkpoint`` records
    the recipe, so the model comes back the way it was built rather than the way a
    reader guessed.
    """
    return Checkpoint.load(checkpoint).model()
