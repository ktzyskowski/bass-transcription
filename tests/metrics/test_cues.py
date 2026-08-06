"""score_cues: pool an encoder's Cues channels and score whichever are supplied.

Pure over aligned (pred, truth) pairs -- no encoder, no audio -- the same core the
offline harness and the async validator both call, so this pins it once.
"""

import numpy as np

from stemify.core.metrics import AlignedPairs, score_cues


def _names(results):
    return {r.name for r in results}


def _perfect_onset():
    true = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    return (true.copy(), true)  # a perfect curve == its own target


def test_all_three_channels_score_when_supplied():
    pred_true = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    hz = np.array([0, 110.0, 110.0, 110.0, 0, 0])
    track = AlignedPairs(
        onset=_perfect_onset(),
        voicing=(pred_true.copy(), pred_true),
        pitch=(hz.copy(), hz),
    )
    results = score_cues([track])
    assert _names(results) == {"onset_pr", "voicing_auroc", "pitch-accuracy"}
    by = {r.name: r for r in results}
    assert by["onset_pr"].scalars["onset_ap"] == 1.0
    assert by["pitch-accuracy"].scalars["rpa"] == 1.0


def test_voicing_omitted_without_a_head():
    results = score_cues([AlignedPairs(onset=_perfect_onset())])
    assert "voicing_auroc" not in _names(results)
    assert "onset_pr" in _names(results)


def test_pitch_omitted_without_a_reference_contour():
    v = np.array([0, 1, 1, 0, 0, 0], dtype=float)
    results = score_cues([AlignedPairs(onset=_perfect_onset(), voicing=(v.copy(), v))])
    assert "pitch-accuracy" not in _names(results)
    assert {"onset_pr", "voicing_auroc"} <= _names(results)


def test_channel_scored_only_when_every_track_supplies_it():
    # One track has voicing, another does not -> voicing is not pooled (never mixes
    # present and absent into one silently-partial score).
    v = np.array([0, 1, 1, 0, 0, 0], dtype=float)
    with_v = AlignedPairs(onset=_perfect_onset(), voicing=(v.copy(), v))
    without_v = AlignedPairs(onset=_perfect_onset())
    assert "voicing_auroc" not in _names(score_cues([with_v, without_v]))


def test_empty_input_scores_nothing():
    assert score_cues([]) == []
