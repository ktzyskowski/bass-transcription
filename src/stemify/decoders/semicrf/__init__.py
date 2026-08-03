"""The semi-CRF decoder, split into testable units.

A learned, global note decoder over per-frame ``Cues`` (contrast the greedy
``GreedyDecoder``). The package composes:

  - ``constants`` -- grid stride, segment labels, duration edges.
  - ``tensor_cues`` -- ``TensorCues``, the training-side twin of ``Cues``, and the
    one-device rule its fields obey.
  - ``cumsums`` -- the prefix sums that make a span score O(1) in its length. They
    change when a span *feature* is added or retired.
  - ``evidence`` -- ``Evidence`` and ``build_evidence``: the per-frame tensors the scan
    reads, always batched, plus the span accessors that own the index conventions.
  - ``spans`` -- ``SpanScorer`` (what a span is worth) and ``Grid``. The *only* place a
    potential is written down; ``LinearSpanScorer`` is the classical log-linear form.
  - ``window`` -- the rolling ``alpha`` buffer both recurrences scan over.
  - ``forward`` -- ``step_terms``, the recurrence itself, and its logsumexp reduction:
    whole-track batched and chunked-with-carry (truncated BPTT).
  - ``viterbi`` -- the same ``step_terms`` under a max reduction, with backpointers.
  - ``gold`` -- *which* segmentation is gold, from reference notes (the audible-gap
    rule). Pure numpy.
  - ``score`` -- what a *given* segmentation is worth. The gold half of the NLL, but
    segmentation-agnostic.
  - ``decoder`` -- ``SemiCRFDecoder``, the ``Decoder`` façade wiring them.

``gold`` and ``score`` are split because they change for different reasons: the first
when the labelling rule changes, the second when the scorer does.

Public surface: ``SemiCRFDecoder``, ``TensorCues``, and ``SpanScorer`` for anyone
implementing a new potential.
"""

from stemify.decoders.semicrf.decoder import SemiCRFDecoder
from stemify.decoders.semicrf.tensor_cues import TensorCues
from stemify.decoders.semicrf.spans import Grid, LinearSpanScorer, SpanScorer

__all__ = [
    "Grid", "LinearSpanScorer", "SemiCRFDecoder", "SpanScorer", "TensorCues",
]
