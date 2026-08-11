"""The semi-CRF: the algorithm, and in ``fit/`` the way it is trained.

A learned, global note decoder over per-frame ``Frames`` (contrast the greedy
``GreedyDecoder``). One package rather than a decode half under ``core`` and a training
half under ``training``: the two were split by *layer* and reached across the seam into
each other's internals -- ``gold``, ``window``, ``constants`` and ``spans`` were all
imported from the training side -- which is the sign that the seam was in the wrong
place. Sliced vertically instead, those are ordinary imports within one package and the
surface facing the rest of the project is two names: ``SemiCRFDecoder`` (below) and
``SemiCRFRecipe`` (in ``fit``).

Everything outside ``fit`` is the algorithm and has no training dependency, so
inference imports this package without pulling in a dataset, a loss or a loader. The
package composes:

  - ``constants`` -- grid stride, segment labels, duration edges.
  - ``placement`` -- the one-device rule the recurrence's arithmetic depends on, and
    the ``as_tensor`` placement that lets one DP read a numpy ``Frames`` as constants
    and a grad-bearing one as graph nodes.
  - ``cumsums`` -- the prefix sums that make a span score O(1) in its length. They
    change when a span *feature* is added or retired.
  - ``evidence`` -- ``Evidence`` and ``build_evidence``: the per-frame tensors the scan
    reads, always batched, plus the span accessors that own the index conventions.
  - ``spans`` -- ``SpanScorer``: what a span is worth, and the *only* place a potential
    is written down. ``neural`` is the one implementation in the tree.
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

Public surface: ``SemiCRFDecoder``, ``SpanScorer`` for anyone
implementing a new potential, and ``Semiring`` for anyone implementing a new reduction.
"""

from stemify.semicrf.decoder import SemiCRFDecoder
from stemify.semicrf.neural import NeuralSpanScorer
from stemify.semicrf.semiring import LOG, MAX_PLUS, Semiring
from stemify.semicrf.spans import SpanScorer

__all__ = [
    "LOG", "MAX_PLUS", "NeuralSpanScorer", "SemiCRFDecoder", "Semiring", "SpanScorer",
]
