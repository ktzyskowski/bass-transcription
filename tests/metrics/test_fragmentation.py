"""Fragmentation: the false-positive breakdown.

Each test isolates one bucket so a miscategorised phantom fails a named test
rather than a single opaque total. The cross-track test is the one that guards
the property the whole note-level design rests on: a phantom in one track must
never be explained by a reference note in another.
"""

from stemify.metrics.note.fragmentation import Fragmentation


def frag(refs, ests):
    return Fragmentation().compute(refs, ests).scalars


def test_fragments_of_a_held_note(note):
    # One held note 0-2 s. The estimate's first piece matches (TP); the two later
    # pieces sit inside the note at the same pitch -> fragmentation.
    ref = [note(0.0, 2.0, 40)]
    est = [note(0.0, 0.5, 40), note(0.6, 1.0, 40), note(1.2, 1.6, 40)]
    s = frag([ref], [est])
    assert s == {"fp": 2, "fragmentation": 2, "interior_offpitch": 0, "rest": 0}


def test_note_in_a_rest_is_voicing(note):
    # A phantom well after the only reference note ends -> a rest, i.e. voicing.
    ref = [note(0.0, 0.5, 40)]
    est = [note(0.0, 0.5, 40), note(2.0, 2.5, 45)]
    s = frag([ref], [est])
    assert s == {"fp": 1, "fragmentation": 0, "interior_offpitch": 0, "rest": 1}


def test_wrong_pitch_inside_a_note_is_interior_offpitch(note):
    # Phantom inside the held note's span but an octave off -> interior_offpitch,
    # counted apart so it inflates neither fragmentation nor voicing.
    ref = [note(0.0, 2.0, 40)]
    est = [note(0.0, 0.5, 40), note(0.6, 1.0, 52)]
    s = frag([ref], [est])
    assert s == {"fp": 1, "fragmentation": 0, "interior_offpitch": 1, "rest": 0}


def test_matched_estimates_are_not_false_positives(note):
    ref = [note(0.0, 1.0, 40), note(1.0, 2.0, 45)]
    est = [note(0.0, 1.0, 40), note(1.0, 2.0, 45)]
    s = frag([ref], [est])
    assert s["fp"] == 0


def test_phantom_is_scoped_to_its_own_track(note):
    # Track A holds note 40 from 0-2 s and is scored perfectly. Track B emits a
    # phantom at the *same time and pitch* as A's held note. It must fall in B's
    # rest bucket, never be excused as a fragment of A.
    a_ref = [note(0.0, 2.0, 40)]
    a_est = [note(0.0, 2.0, 40)]
    b_ref = []
    b_est = [note(0.5, 1.0, 40)]
    s = frag([a_ref, b_ref], [a_est, b_est])
    assert s == {"fp": 1, "fragmentation": 0, "interior_offpitch": 0, "rest": 1}


def test_buckets_sum_to_false_positives(note):
    # One of each kind, plus a matched note, across two tracks.
    ref = [note(0.0, 2.0, 40)]
    est = [
        note(0.0, 0.5, 40),   # TP
        note(0.6, 1.0, 40),   # fragmentation
        note(1.2, 1.6, 52),   # interior_offpitch
        note(5.0, 5.5, 45),   # rest
    ]
    s = frag([ref], [est])
    assert s["fp"] == s["fragmentation"] + s["interior_offpitch"] + s["rest"]
    assert s == {"fp": 3, "fragmentation": 1, "interior_offpitch": 1, "rest": 1}


def test_result_name_and_render(note):
    result = Fragmentation().compute(
        [[note(0.0, 2.0, 40)]], [[note(0.0, 0.5, 40), note(0.6, 1.0, 40)]]
    )
    assert result.name == "fragmentation"
    assert "fragmentation" in result.render()
