"""The layering, as a test rather than a convention.

Two properties, both of which were true by inspection when the layers were drawn and
neither of which stays true on its own. A structural rule nobody can violate by accident
is worth more than a docstring saying not to: the failure mode being guarded is a single
convenient import that re-couples two layers, which reads as a one-line diff and is
invisible in review.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "stemify"

# What each layer may import. The DAG, stated once.
#
# ``core`` is being dissolved into vertical pillars, bottom-up. Each extraction adds a
# row here *before* the move, so the new pillar's edges are policed from the moment it
# exists rather than after it has grown some.
#
# A pillar's row is *exact*: it names the edges that actually exist, not the ones it
# would be entitled to. See ``test_a_pillars_allowance_is_not_aspirational``.
PILLAR_ALLOWANCE = {
    "platform": set(),
    "music": set(),
    "persistence": {"platform", "music"},
    "audio": {"platform", "persistence"},
    "corpus": {"music", "audio"},
    "transcription": {"music", "audio", "persistence"},
}
_BELOW = set(PILLAR_ALLOWANCE)
ALLOWED = {
    **PILLAR_ALLOWANCE,
    "core": _BELOW,
    "models": _BELOW | {"core"},
    # `fitting` holds what a training *strategy* implements -- Recipe, Loss, Batch,
    # TrainConfig. It sits above `models` because `Recipe.build_model()` returns a
    # FrameNet, which is precisely why it could not be merged with the inference
    # contracts in `transcription`: those sit *below* models, which implements them.
    "fitting": _BELOW | {"core", "models"},
    # An extension: it implements `fitting` and knows nothing of the framework that
    # drives it. This used to import `training` and be imported by it.
    "semicrf": _BELOW | {"core", "models", "fitting"},
    # The framework, and the composition root that names the strategies.
    "training": _BELOW | {"core", "models", "fitting", "semicrf"},
    "evaluation": _BELOW | {"core", "models", "fitting"},
    "serving": _BELOW | {"core", "models", "fitting"},
}

# The pillars extracted from ``core`` so far, in dependency order. A pillar is held to a
# stronger rule than a layer: it may import only the pillars *below* it, which for the
# bottom one means nothing from ``stemify`` at all.
#
# This is the property that makes a pillar testable without doubles, and the reason
# `core` failed it: its `Cache` welded corpus, transform, features and persistence into
# one call, so every training test had to subclass it into a `_NoOpCache` just to get it
# out of the way. A pillar its own consumers must stub is a pillar whose seam is in the
# wrong place. `persistence` is what that inverted into.
#
# The graph is not a chain. `audio` reaches `platform` for the filesystem and the
# accelerator and `persistence` to hold a CQT, but never `music` -- a waveform has
# nothing to say about notes. Only `corpus` needs both, because an item is the place
# audio and truth meet. `persistence` sits *below* `audio` rather than above it, which
# is what lets the CQT memoiser live in `audio` where the CQT is defined instead of in
# a class that has to know about corpora to reach it.
PILLARS = list(PILLAR_ALLOWANCE)

# Edges that exist today and are known to be wrong -- the work still outstanding from
# the layering plan (docs/refactor-plan.md, steps A and C). Listed rather than allowed,
# so the set can only shrink: adding a new one fails this test, and removing the last of
# them is what makes the layer above genuinely independent.
#
#   A  evaluation/serving read checkpoints and name SemiCRFDecoder directly; both go
#      away with `Transcriber.from_checkpoint` and the decoder registry.
#   C  training.cli imports the semi-CRF's weight constants and AUX_TERMS by name;
#      goes away when a recipe contributes its own argument group.
#
# `("training", "semicrf")` used to be here as half of a genuine cycle. It is now a
# legal downward edge: `training` drives the strategies, `semicrf` is one of them.
PENDING = {
    ("evaluation", "training"),
    ("evaluation", "semicrf"),
    ("serving", "training"),
    ("serving", "semicrf"),
}

# The semi-CRF is an *extension*. It now implements `fitting` and imports nothing at all
# from `training`, which is what makes the two packages separately detachable.


def _layer(module: str) -> str:
    return module.split(".")[0]


def _edges() -> list[tuple[str, str, str]]:
    """Every internal import, as ``(source module, target module, source file)``."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = str(path.relative_to(SRC)).removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "stemify"
            ):
                target = node.module.removeprefix("stemify").removeprefix(".")
                if target:
                    out.append((module, target, str(path.relative_to(SRC))))
    return out


def test_layer_dependencies_follow_the_dag():
    """No layer imports one above it -- except the edges PENDING already names."""
    violations = []
    for source, target, where in _edges():
        src_layer, dst_layer = _layer(source), _layer(target)
        if src_layer == dst_layer:
            continue
        edge = (src_layer, dst_layer)
        if dst_layer in ALLOWED.get(src_layer, set()) or edge in PENDING:
            continue
        violations.append(f"{where}: {source} -> {target}  ({src_layer} -> {dst_layer})")
    assert not violations, "layer violations:\n  " + "\n  ".join(violations)


@pytest.mark.parametrize("pillar", PILLARS)
def test_a_pillar_depends_only_on_the_pillars_beneath_it(pillar):
    """Detachability, as the property rather than the intention.

    Stronger than the DAG check above in the one way that matters: **no PENDING**. That
    list is how an inherited violation gets tolerated while it is being unwound, and a
    pillar is precisely a package that has none -- it is extracted *because* its edges
    are already clean, so an exemption here would mean the extraction was premature.

    ``platform`` and ``music`` sit at the bottom, so their allowance is empty and this
    asserts they import nothing from ``stemify`` whatsoever.
    """
    beneath = set(ALLOWED[pillar])
    assert beneath <= set(PILLARS), f"`{pillar}` may only be allowed other pillars"
    strays = [
        f"{where}: {source} -> {target}"
        for source, target, where in _edges()
        if _layer(source) == pillar and _layer(target) not in beneath | {pillar}
    ]
    assert not strays, (
        f"`{pillar}` may import only {sorted(beneath) or 'nothing'}, but:\n  "
        + "\n  ".join(strays)
    )


@pytest.mark.parametrize("pillar", PILLARS)
def test_a_pillars_allowance_is_not_aspirational(pillar):
    """``PILLAR_ALLOWANCE`` names the edges that exist, not the ones a pillar could have.

    The same discipline ``PENDING`` is held to, for the same reason: a declared graph
    that drifts from the real one stops being a description and starts being a licence.
    An entry nobody uses is a claim of coupling that is not there -- and the next
    genuinely-new edge then slips in under it, unremarked, because the row already
    permitted it.

    Deleting a line is the fix. That deletion is the signal: it means a pillar just got
    more detachable, which is the thing this whole exercise is for.
    """
    used = {
        _layer(target)
        for source, target, _ in _edges()
        if _layer(source) == pillar and _layer(target) != pillar
    }
    unused = set(PILLAR_ALLOWANCE[pillar]) - used
    assert not unused, (
        f"`{pillar}` is allowed {sorted(unused)} but imports nothing from them. "
        "Delete the entry."
    )


def test_pending_edges_are_all_still_real():
    """PENDING may only shrink. An entry that no longer occurs must be deleted, or the
    list stops describing the work and starts hiding it."""
    seen = {(_layer(s), _layer(t)) for s, t, _ in _edges() if _layer(s) != _layer(t)}
    assert not (PENDING - seen), (
        f"PENDING lists edges that no longer exist: {sorted(PENDING - seen)}. "
        "Delete them -- that step of the refactor is done."
    )


def test_semicrf_does_not_import_training_at_all():
    """An extension depends on the contracts it implements, never on the framework that
    drives it, and never on a sibling extension.

    This asserted a *permitted set* while `semicrf` still named `training.base` and
    `training.config` for its contracts, plus one known exception: `SemiCRFRecipe`
    constructed a whole `FrameRecipe` just to call its `build_model`, because assembling
    the encoder had no other home. The contracts moved to `fitting` and the assembly
    moved to `models.assembly`, so the set is now empty and the rule is the simple one.
    """
    bad = [
        f"{where}: {source} -> {target}"
        for source, target, where in _edges()
        if _layer(source) == "semicrf" and _layer(target) == "training"
    ]
    assert not bad, "semicrf reached into training:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("package", [*PILLARS, "core"])
def test_the_lower_layers_import_no_torch(package):
    """Importing every module in ``package`` must not pull torch in.

    In a subprocess because the test session has torch loaded already, and by walking
    the package rather than importing it alone -- every ``__init__`` here is a docstring,
    so importing one proves nothing about the modules underneath.
    """
    probe = f"""
import importlib, pkgutil, sys
import stemify.{package} as pkg
for m in pkgutil.walk_packages(pkg.__path__, "stemify.{package}."):
    importlib.import_module(m.name)
torch = sorted(n for n in sys.modules if n == "torch" or n.startswith("torch."))
if torch:
    raise SystemExit("{package} pulled in torch: " + ", ".join(torch[:5]))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    if result.returncode:
        pytest.fail(
            f"{result.stdout.strip()}{result.stderr.strip()}\n\n"
            "Deep learning belongs in `stemify.models`. A torch import down here "
            "means a contract has been given a neural implementation in place."
        )
