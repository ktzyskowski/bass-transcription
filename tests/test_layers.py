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
ALLOWED = {
    "music": set(),
    "core": {"music"},
    "models": {"music", "core"},
    "training": {"music", "core", "models"},
    "semicrf": {"music", "core", "models", "training"},
    "evaluation": {"music", "core", "models"},
    "serving": {"music", "core", "models"},
}

# The pillars extracted from ``core`` so far, in dependency order. A pillar is held to a
# stronger rule than a layer: it may import only the pillars *below* it, which for the
# bottom one means nothing from ``stemify`` at all.
#
# This is the property that makes a pillar testable without doubles. ``core`` failed it
# in both directions -- ``Cache`` welds corpus, transform, features and persistence into
# one call, which is why every training test has to subclass it into a ``_NoOpCache``
# just to get it out of the way. A pillar that its own consumers must stub is a pillar
# whose seam is in the wrong place.
PILLARS = ["music"]

# Edges that exist today and are known to be wrong -- the work still outstanding from
# the layering plan (docs/refactor-plan.md, steps A and C). Listed rather than allowed,
# so the set can only shrink: adding a new one fails this test, and removing the last of
# them is what makes the layer above genuinely independent.
#
#   A  evaluation/serving read checkpoints and name SemiCRFDecoder directly; both go
#      away with `Transcriber.from_checkpoint` and the decoder registry.
#   C  training.cli imports the semi-CRF's weight constants and AUX_TERMS by name;
#      goes away when a recipe contributes its own argument group.
PENDING = {
    ("evaluation", "training"),
    ("evaluation", "semicrf"),
    ("serving", "training"),
    ("serving", "semicrf"),
    ("training", "semicrf"),
}

# The semi-CRF is an *extension*: it implements training's contracts, so it may name
# them -- and nothing else in that package. This is what stops a plugin reaching into a
# sibling plugin (it built a `FrameRecipe` to reuse encoder assembly, which is step D).
SEMICRF_MAY_IMPORT_FROM_TRAINING = {"training.base", "training.config"}


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

    Stricter than the DAG above, which only forbids importing *upward*: a pillar may not
    reach sideways either. ``music`` is at the bottom, so its allowance is empty and this
    asserts it imports nothing from ``stemify`` whatsoever.
    """
    beneath = set(PILLARS[: PILLARS.index(pillar)])
    strays = [
        f"{where}: {source} -> {target}"
        for source, target, where in _edges()
        if _layer(source) == pillar and _layer(target) not in beneath | {pillar}
    ]
    assert not strays, (
        f"`{pillar}` may import only {sorted(beneath) or 'nothing'}, but:\n  "
        + "\n  ".join(strays)
    )


def test_pending_edges_are_all_still_real():
    """PENDING may only shrink. An entry that no longer occurs must be deleted, or the
    list stops describing the work and starts hiding it."""
    seen = {(_layer(s), _layer(t)) for s, t, _ in _edges() if _layer(s) != _layer(t)}
    assert not (PENDING - seen), (
        f"PENDING lists edges that no longer exist: {sorted(PENDING - seen)}. "
        "Delete them -- that step of the refactor is done."
    )


def test_semicrf_touches_only_the_training_contracts():
    """An extension depends on the framework, never on a sibling extension."""
    bad = [
        f"{where}: {source} -> {target}"
        for source, target, where in _edges()
        if _layer(source) == "semicrf"
        and _layer(target) == "training"
        and target not in SEMICRF_MAY_IMPORT_FROM_TRAINING
    ]
    # `semicrf.fit.recipe` builds a `FrameRecipe` purely to reuse encoder assembly.
    # Step D moves that into `models`; until then this is the one known exception.
    expected = ["semicrf/fit/recipe.py: semicrf.fit.recipe -> training.frame.recipe"]
    assert bad == expected, f"unexpected semicrf -> training edges: {bad}"


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
