"""The semi-CRF decoder, split into testable units.

A learned, global note decoder over per-frame ``Cues`` (contrast the greedy
``HeuristicDecoder``). The package composes:

  - ``constants`` -- grid stride, segment labels, duration edges.
  - ``emissions`` -- ``TensorCues`` and ``Cues|TensorCues -> Emissions`` (the tensors
    the scan reads).
  - ``potentials`` -- ``SemiCRFPotentials``, the ~15 learned scalars + unary/transition
    scores.
  - ``gold`` -- the gold segmentation from reference notes (the audible-gap rule).
  - ``dp`` -- the segmental forward (``log Z``) and Viterbi recurrences.
  - ``decoder`` -- ``SemiCRFDecoder``, the ``Decoder`` façade wiring them.

Public surface: ``SemiCRFDecoder`` and ``TensorCues``.
"""

from stemify.decoders.semicrf.decoder import SemiCRFDecoder
from stemify.decoders.semicrf.emissions import TensorCues

__all__ = ["SemiCRFDecoder", "TensorCues"]
