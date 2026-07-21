"""The shared note matcher -- every note-level number is built on this.

If ``matched_pairs`` is wrong, PRF and Fragmentation are both wrong in lockstep,
so these are the load-bearing tests. They pin the four properties the metrics
rely on: onset tolerance, pitch tolerance, offsets ignored, and one-match-per-
reference.
"""

from stemify.evaluation.metrics.note.base import (
    ONSET_TOLERANCE_S,
    PITCH_TOLERANCE_CENTS,
    count_matches,
    matched_pairs,
)


def test_exact_match(note):
    assert matched_pairs([note(0.0, 1.0, 40)], [note(0.0, 1.0, 40)]) == [(0, 0)]


def test_onset_within_tolerance_matches(note):
    # 40 ms < 50 ms tolerance -> still the same note
    assert matched_pairs([note(0.0, 1.0, 40)], [note(0.04, 1.0, 40)]) == [(0, 0)]


def test_onset_beyond_tolerance_does_not_match(note):
    # 200 ms late -> a different note
    assert matched_pairs([note(0.0, 1.0, 40)], [note(0.2, 1.0, 40)]) == []


def test_pitch_beyond_tolerance_does_not_match(note):
    # a whole semitone (100 cents) off -> wrong pitch, no match
    assert matched_pairs([note(0.0, 1.0, 40)], [note(0.0, 1.0, 41)]) == []


def test_offsets_are_ignored(note):
    # Same onset and pitch, wildly different offset (0.1 s vs 5 s). Bass offsets
    # are fuzzy, so the matcher must not care -- this is offset_ratio=None.
    assert matched_pairs([note(0.0, 5.0, 40)], [note(0.0, 0.1, 40)]) == [(0, 0)]


def test_reference_matches_at_most_one_estimate(note):
    # Two estimates both land on the one reference note; only one may claim it.
    pairs = matched_pairs([note(0.0, 1.0, 40)], [note(0.0, 1.0, 40), note(0.02, 1.0, 40)])
    assert len(pairs) == 1
    assert [rp for rp, _ in pairs] == [0]  # the single reference, matched once


def test_empty_sides(note):
    assert matched_pairs([], [note(0.0, 1.0, 40)]) == []
    assert matched_pairs([note(0.0, 1.0, 40)], []) == []
    assert matched_pairs([], []) == []


def test_default_tolerances_are_the_protocol():
    assert ONSET_TOLERANCE_S == 0.05
    assert PITCH_TOLERANCE_CENTS == 50.0


def test_count_matches_tp_fp_fn(note):
    ref = [note(0.0, 1.0, 40), note(1.0, 2.0, 45)]
    est = [note(0.0, 1.0, 40), note(1.0, 2.0, 45), note(3.0, 3.5, 50)]  # +1 phantom
    assert count_matches(ref, est) == (2, 1, 0)


def test_count_matches_empty_sides(note):
    assert count_matches([], [note(0.0, 1.0, 40), note(2.0, 3.0, 41)]) == (0, 2, 0)
    assert count_matches([note(0.0, 1.0, 40)], []) == (0, 0, 1)
    assert count_matches([], []) == (0, 0, 0)