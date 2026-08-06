"""The frame-family scorer: pool an encoder's ``Cues`` channels across tracks.

The peer of ``metrics.note.scoring`` (note pooling), for the frame family. ``Cues``
is the encoder's per-frame output: an onset curve, a voicing curve, and a pitch
contour. Scoring it -- decoder-independent -- is the encoder altitude of the pipeline.
This module is the *pure* core: it takes, per track, the already-aligned
``(prediction, truth)`` pair for each channel and pools frames across tracks (the
``FrameMetric`` contract: concatenation is lossless, so one ``compute`` over the
pool is frame-weighted by construction).

It lives in ``metrics`` rather than ``evaluation`` on purpose: both the offline
``CuesEvaluation`` harness and the in-training async validator score through it, and
the validator is in ``training`` -- routing it through ``evaluation`` would invert the
dependency. So the core stays a leaf that depends only on the metrics beside it, and
each caller does its own alignment (building the target grids, resampling the contour)
before handing pairs here. A channel whose pair is ``None`` is simply not scored:
onset is always available, voicing needs a voicing head, pitch needs a reference
contour (the offline path supplies it; the validator omits it, since pyin pitch does
not move between checkpoints).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stemify.core.metrics.base import MetricResult
from stemify.core.metrics.frame.onset import OnsetPR
from stemify.core.metrics.frame.pitch import PitchAccuracy
from stemify.core.metrics.frame.voicing import VoicingAUROC

# A per-channel aligned pair: (prediction, truth), both 1-D and the same length. For
# onset/voicing that is (probability, label); for pitch (estimate Hz, reference Hz).
Pair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class AlignedPairs:
    """One track's aligned ``(prediction, truth)`` pairs, ready to pool and score.

    ``onset`` is always present. ``voicing`` is ``None`` for an encoder with no
    voicing head; ``pitch`` is ``None`` when no reference contour is available. The
    caller has already put every pair on a common grid -- this module only pools."""

    onset: Pair
    voicing: Pair | None = None
    pitch: Pair | None = None


def _pool(pairs: list[Pair]) -> Pair:
    """Concatenate per-track ``(pred, truth)`` pairs into one pooled pair."""
    preds = np.concatenate([p for p, _ in pairs])
    truths = np.concatenate([t for _, t in pairs])
    return preds, truths


def score_cues(tracks: Sequence[AlignedPairs]) -> list[MetricResult]:
    """Score whichever channels every track supplies, pooled across tracks.

    Pure given ``tracks`` -- no encoder, no dataset, no audio -- so the offline harness
    and the async validator score identically. A channel is scored only when *every*
    track carries it, so a pooled score never silently mixes present and absent."""
    if not tracks:
        return []

    results: list[MetricResult] = []

    onset_pred, onset_true = _pool([t.onset for t in tracks])
    results.append(OnsetPR().compute(onset_true, onset_pred))

    # Filter-then-count rather than `all(... is not None)` then re-read: the two
    # express the same "every track carries it" rule, but this one leaves the list
    # holding the non-optional type, so the pooled call cannot be handed a None that
    # a later edit to the guard stopped excluding.
    voicings = [t.voicing for t in tracks if t.voicing is not None]
    if len(voicings) == len(tracks):
        voi_pred, voi_true = _pool(voicings)
        results.append(VoicingAUROC().compute(voi_true, voi_pred))

    pitches = [t.pitch for t in tracks if t.pitch is not None]
    if len(pitches) == len(tracks):
        pitch_est, pitch_ref = _pool(pitches)
        results.append(PitchAccuracy().compute(pitch_ref, pitch_est))

    return results
