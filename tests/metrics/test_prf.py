"""PRF: precision / recall / F1 pooled across tracks.

The one property worth guarding hardest is that pooling sums *counts* and
derives rates once -- never averages per-track rates. ``test_pooling_is_count_
weighted_not_track_averaged`` is the case that fails loudly if someone
"simplifies" it into a mean of per-track F1.
"""

import math

from stemify.metrics.note.prf import PRF


def test_single_track(note):
    refs = [[note(0.0, 1.0, 40), note(1.0, 2.0, 45)]]
    ests = [[note(0.0, 1.0, 40), note(1.0, 2.0, 45), note(3.0, 3.5, 50)]]  # +1 phantom
    s = PRF().compute(refs, ests).scalars
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 0)
    assert s["precision"] == 2 / 3
    assert s["recall"] == 1.0
    assert math.isclose(s["f1"], 0.8)


def test_pooling_sums_counts_across_tracks(note):
    refs = [[note(0.0, 1.0, 40)], [note(0.0, 1.0, 50)]]
    ests = [[note(0.0, 1.0, 40)], [note(0.0, 1.0, 50), note(2.0, 2.5, 55)]]
    s = PRF().compute(refs, ests).scalars
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 0)


def test_per_track_counts_are_emitted_and_sum_to_the_pooled_totals(note):
    # The worst-item ranking reads these instead of a second match, so the rows must
    # be in input order and sum (column-wise) to the pooled tp/fp/fn.
    refs = [[note(0.0, 1.0, 40)], [note(0.0, 1.0, 50)]]
    ests = [[note(0.0, 1.0, 40)], [note(0.0, 1.0, 50), note(2.0, 2.5, 55)]]
    result = PRF().compute(refs, ests)
    per_item = result.arrays["per_item_counts"]
    assert per_item.shape == (2, 3)
    assert list(per_item[0]) == [1, 0, 0]   # track 0: one true positive
    assert list(per_item[1]) == [1, 1, 0]   # track 1: a hit plus one phantom
    s = result.scalars
    assert list(per_item.sum(axis=0)) == [s["tp"], s["fp"], s["fn"]]


def test_empty_dataset_has_no_per_track_rows():
    assert PRF().compute([], []).arrays["per_item_counts"].shape == (0, 3)


def test_pooling_is_count_weighted_not_track_averaged(note):
    # Track A: 3 reference notes, all found (per-track F1 = 1.0).
    # Track B: 1 reference missed + 1 phantom (per-track F1 = 0.0).
    # Averaging per-track F1 gives 0.5; pooling counts gives tp3/fp1/fn1 -> 0.75.
    # The pooled answer is the correct, note-weighted one.
    a_ref = [note(0.0, 1.0, 40), note(1.0, 2.0, 41), note(2.0, 3.0, 42)]
    a_est = list(a_ref)
    b_ref = [note(0.0, 1.0, 50)]
    b_est = [note(5.0, 5.5, 55)]  # wrong note, real one missed
    s = PRF().compute([a_ref, b_ref], [a_est, b_est]).scalars
    assert (s["tp"], s["fp"], s["fn"]) == (3, 1, 1)
    assert s["precision"] == 0.75
    assert s["recall"] == 0.75
    assert s["f1"] == 0.75


def test_empty_dataset_is_zero_not_crash():
    s = PRF().compute([], []).scalars
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 0)
    assert (s["precision"], s["recall"], s["f1"]) == (0.0, 0.0, 0.0)


def test_all_phantoms_zero_precision(note):
    s = PRF().compute([[]], [[note(0.0, 1.0, 40)]]).scalars
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 0)
    assert s["precision"] == 0.0


def test_result_name_and_render(note):
    result = PRF().compute([[note(0.0, 1.0, 40)]], [[note(0.0, 1.0, 40)]])
    assert result.name == "note-prf"
    assert "F1" in result.render()
