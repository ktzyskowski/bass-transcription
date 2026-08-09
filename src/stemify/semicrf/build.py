"""``build_scorer``: a scorer's recorded config -> the scorer.

Separate from ``spans.py`` only because it has to know every implementation while the
contract must know none -- ``neural`` imports ``spans``, so a factory living there would
close the cycle. It is dispatch and nothing else; each scorer owns both directions of
its own serialisation, next to the ``__init__`` they have to agree with.

Dispatch is on the recorded ``kind`` rather than on which keys are present. A config is
data off disk, and "whatever has ``dur_edges_s`` is linear" is the kind of rule that
quietly rebuilds the wrong model once a second scorer grows a similar field.
"""

from collections.abc import Mapping
from typing import Any

from stemify.semicrf.neural import NeuralSpanScorer
from stemify.semicrf.spans import LinearSpanScorer, SpanScorer

SCORERS: dict[str, type[SpanScorer]] = {
    "linear": LinearSpanScorer,
    "neural": NeuralSpanScorer,
}


def build_scorer(config: Mapping[str, Any]) -> SpanScorer:
    """Rebuild the scorer a checkpoint was trained with, weights not yet loaded.

    ``kind`` defaults to ``linear`` for checkpoints written before scorers recorded one.
    """
    kind = config.get("kind", "linear")
    if kind not in SCORERS:
        raise ValueError(
            f"unknown scorer kind {kind!r}; have {sorted(SCORERS)}. A checkpoint from a "
            "revision carrying a scorer this one does not cannot be rebuilt -- its "
            "weights describe a model that no longer exists."
        )
    return SCORERS[kind].from_config(config)
