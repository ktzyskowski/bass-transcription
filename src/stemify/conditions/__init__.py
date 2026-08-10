"""The experimental arm: what audio a model is shown, over which tracks.

A ``Condition`` is the thing an ablation varies. It names the ordered channels the model
reads -- each a ``Sources`` role seen through one ``AudioTransform`` -- and the track
filter that goes with them, which is what makes two runs comparable or not.

    types     ``Channel`` and ``Condition``: the shape of an input
    registry  ``CONDITIONS``, ``resolve``, ``canonical``: which ones exist, by tag

Two modules because they change for different reasons. The types change when the
*shape* does -- a new role, a new transform hook -- while the registry changes whenever
an experiment wants a new combination, which is far more often.

**Why a pillar and not part of ``training``.** The evaluation harness and every analysis
script vary conditions too, and a run tag recorded in ``docs/experiments.md`` has to
resolve identically from all three. ``LEGACY_NAMES`` is the proof: renamed tags still
resolve, with a warning, because a rename that silently invalidates every documented
experiment is worse than the names it fixes.

**The filter is the load-bearing part**, and the least obvious. ``unambiguous_only``
drops multi-bass tracks the moment a mixture reaches the model, because one reference
contour cannot describe two basses sounding at once -- the model would be shown both and
scored against one. Getting that wrong does not raise; it changes the corpus underneath
an A/B, which is how a 0.878 once looked worse than a 0.9375 that was mostly an easier
subset.

Below it: ``corpus`` for the track set, ``audio`` for the transforms and the CQT cache,
``persistence`` for where that cache lives.
"""

from stemify.conditions.registry import (
    CLEAN_BY_WIDTH,
    CONDITIONS,
    LEGACY_NAMES,
    Channel,
    Condition,
    canonical,
    resolve,
)

__all__ = [
    "CLEAN_BY_WIDTH", "CONDITIONS", "LEGACY_NAMES",
    "Channel", "Condition", "canonical", "resolve",
]
