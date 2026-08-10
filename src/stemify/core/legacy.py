"""Readers for artifacts written by an older version of this code. One place.

**Why this is nearly empty, and should stay that way.** The artifacts that cannot be
regenerated cheaply are checkpoints, the feature/contour cache, MLflow history and the
commands pinned in ``docs/ablation-plan.md``. None of them records a *module path*:
``cache.content_key`` hashes ``AudioTransform.name``, ``features.SIGNATURE`` and
resolved audio paths, and a checkpoint payload is ``{version, recipe: {name, config},
model_state, objective_state}`` where every config value is a primitive. So code can be
moved, split and renamed freely and every cached artifact still resolves -- *provided no
string changes*. Relocation never needs an entry here. Only a deliberate rename does,
which makes each entry below a decision someone made rather than fallout.

**House style, from the two migrations that came before this module.** Every entry is
(a) versioned or idempotent, so running it twice is safe and a half-finished run is
recoverable; (b) loud about what it inferred, because a silent guess about an old
artifact produces numbers from a model nobody trained; (c) documented with the condition
that lets it be *deleted*. A legacy layer with no exit condition is just the old code
with a worse name.

Not yet here, and deliberately: ``CONDITION_ALIASES`` and ``RECIPE_ALIASES`` (old tag ->
current key). They belong at the registry lookups, and those registries do not exist
yet -- declaring empty tables nothing reads would be machinery pretending to be a
contract. They land with the registries, as does ``Checkpoint._from_v1``, which is the
other existing migration and stays beside the format it reads until that format moves.
"""

from pathlib import Path

import numpy as np

from stemify.persistence.serde import features_npy


def migrate_example_cache(cache_root: Path) -> int:
    """Rewrite legacy npz (features stored inside) to the split format: features to a
    bare .npy sidecar, the rest kept in a slim npz. No CQT recompute -- just reads
    each npz and re-lays-it-out. Idempotent: already-split files are skipped.
    Returns the number migrated. One-off: droppable once every cache is split."""
    migrated = 0
    for npz in sorted(cache_root.rglob("*.npz")):
        with np.load(npz, allow_pickle=False) as z:
            if "features" not in z.files:  # already split
                continue
            features = z["features"]
            rest = {k: z[k] for k in z.files if k != "features"}
        np.save(features_npy(npz), features)
        np.savez_compressed(npz, **rest)  # overwrite without the features array
        migrated += 1
    return migrated
