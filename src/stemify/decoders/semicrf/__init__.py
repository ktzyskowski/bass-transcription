"""The semi-CRF decoder, split into testable units.

A learned, global note decoder over per-frame ``Cues`` (contrast the greedy
``HeuristicDecoder``). The package composes:

  - ``constants`` -- grid stride, segment labels, duration edges.
  - ``emissions`` -- ``TensorCues`` and ``Cues|TensorCues -> Emissions`` (the tensors
    the scan reads).
  - ``potentials`` -- ``SemiCRFPotentials``, the ~15 learned scalars + unary/transition
    scores.
  - ``window`` -- the rolling ``alpha`` buffer both recurrences scan over.
  - ``forward`` -- ``log Z`` in the logsumexp semiring: whole-track, batched, and
    chunked-with-carry (truncated BPTT).
  - ``viterbi`` -- the same recurrence in the max semiring, with backpointers.
  - ``gold`` -- *which* segmentation is gold, from reference notes (the audible-gap
    rule). Pure numpy.
  - ``score`` -- what a *given* segmentation is worth under the potentials. The gold
    half of the NLL, but segmentation-agnostic.
  - ``decoder`` -- ``SemiCRFDecoder``, the ``Decoder`` façade wiring them.

``gold`` and ``score`` are split because they change for different reasons: the first
when the labelling rule changes, the second when the potentials do.

Public surface: ``SemiCRFDecoder`` and ``TensorCues``.
"""

from stemify.decoders.semicrf.decoder import SemiCRFDecoder
from stemify.decoders.semicrf.emissions import TensorCues

__all__ = ["SemiCRFDecoder", "TensorCues"]
