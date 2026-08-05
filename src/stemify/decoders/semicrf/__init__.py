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
  - ``window`` -- the rolling ``alpha`` buffer every scan slides over.
  - ``semiring`` -- ``LOG`` and ``MAX_PLUS``: *how* a step's alternatives combine, and
    the only thing separating training's answer from decoding's.
  - ``forward`` -- ``step_terms`` and ``scan``: the recurrence, and the one loop over
    it, whole-track batched or chunked-with-carry (truncated BPTT).
  - ``viterbi`` -- the traceback. The recurrence is ``scan`` under ``MAX_PLUS``; what
    lives here is the part no semiring supplies, since a semiring reduces alternatives
    to a *score* and a score is not a path.
  - ``gold`` -- *which* segmentation is gold, from reference notes (the audible-gap
    rule). Pure numpy.
  - ``score`` -- what a *given* segmentation is worth. The gold half of the NLL, but
    segmentation-agnostic.
  - ``decoder`` -- ``SemiCRFDecoder``, the ``Decoder`` façade wiring them.

``gold`` and ``score`` are split because they change for different reasons: the first
when the labelling rule changes, the second when the scorer does. ``semiring`` and
``spans`` split along the same line the recurrence does: one owns what a segmentation is
*worth*, the other how competing segmentations are *combined*, and a new potential
touches neither the loop nor the algebra.

Public surface: ``SemiCRFDecoder``, ``TensorCues``, ``SpanScorer`` for anyone
implementing a new potential, and ``Semiring`` for anyone implementing a new reduction.
"""

from stemify.decoders.semicrf.decoder import SemiCRFDecoder
from stemify.decoders.semicrf.semiring import LOG, MAX_PLUS, Semiring
from stemify.decoders.semicrf.tensor_cues import TensorCues
from stemify.decoders.semicrf.spans import Grid, LinearSpanScorer, SpanScorer

__all__ = [
    "LOG", "MAX_PLUS", "Grid", "LinearSpanScorer", "SemiCRFDecoder", "Semiring",
    "SpanScorer", "TensorCues",
]
