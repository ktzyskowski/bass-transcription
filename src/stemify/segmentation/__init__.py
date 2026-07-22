"""Segmentation: pitch contour -> discrete notes.

The interface (``Segmenter``) and the DSP floor (``HeuristicSegmenter``) are
exported here. ``LearnedSegmenter`` lives in ``.learned`` and must be imported
explicitly -- it pulls in torch and the trained model, so keeping it out of this
package's ``__init__`` means the light DSP path never transitively imports them.
"""

from stemify.segmentation.base import Segmenter
from stemify.segmentation.heuristic import (
    MERGE_GAP_S,
    MIN_NOTE_S,
    ONSET_DELTA,
    PITCH_SPLIT_SEMITONES,
    HeuristicSegmenter,
    detect_onsets,
)

__all__ = [
    "Segmenter",
    "HeuristicSegmenter",
    "detect_onsets",
    "MIN_NOTE_S",
    "PITCH_SPLIT_SEMITONES",
    "MERGE_GAP_S",
    "ONSET_DELTA",
]
