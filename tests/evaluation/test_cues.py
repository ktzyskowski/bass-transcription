"""CuesEvaluation's pure scoring core: which channels score, and channel omission.

Synthetic Cues + references, no encoder or audio -- the same split the harness and
the async validator both call, so this pins the shared code once.
"""

import numpy as np

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.evaluation.cues import ScoredCues, score_cues
from stemify.notes import NoteEvent
from stemify.training.labels import onset_targets, voicing_targets

FRAME_TIMES = np.arange(12) * 0.0116


def _perfect_track(with_voicing: bool, with_contour: bool) -> ScoredCues:
    notes = [NoteEvent(onset=0.03, offset=0.09, midi_pitch=45.0)]
    onset = onset_targets(np.array([n.onset for n in notes]), FRAME_TIMES)
    ref = Contour(times=FRAME_TIMES, freqs=np.full(FRAME_TIMES.size, 110.0))
    voiced = voicing_targets(ref.times, ref.voiced, FRAME_TIMES)
    return ScoredCues(
        cues=Cues(
            contour=ref if with_contour else Contour(FRAME_TIMES, np.full(12, 110.0)),
            onset=onset,  # a perfect onset curve == its own target
            frame_times=FRAME_TIMES,
            voicing=voiced if with_voicing else None,
        ),
        reference_notes=notes,
        voicing_target=voiced if with_voicing else None,
        reference_contour=ref if with_contour else None,
    )


def _names(results):
    return {r.name for r in results}


def test_all_three_channels_score_when_supplied():
    results = score_cues([_perfect_track(with_voicing=True, with_contour=True)])
    assert _names(results) == {"onset_pr", "voicing_auroc", "pitch-accuracy"}
    by = {r.name: r for r in results}
    assert by["onset_pr"].scalars["onset_ap"] == 1.0
    assert by["pitch-accuracy"].scalars["rpa"] == 1.0


def test_voicing_omitted_without_a_head():
    # An onset-only encoder (cues.voicing is None) -> no voicing channel scored.
    results = score_cues([_perfect_track(with_voicing=False, with_contour=True)])
    assert "voicing_auroc" not in _names(results)
    assert "onset_pr" in _names(results)


def test_pitch_omitted_without_a_reference_contour():
    # The async validator omits the reference contour -> pitch is not scored.
    results = score_cues([_perfect_track(with_voicing=True, with_contour=False)])
    assert "pitch-accuracy" not in _names(results)
    assert {"onset_pr", "voicing_auroc"} <= _names(results)


def test_empty_input_scores_nothing():
    assert score_cues([]) == []
