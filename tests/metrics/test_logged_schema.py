"""The MLflow metric names a run logs, pinned.

These names are an interface, not an implementation detail, and three things depend
on them outside this repo's control flow:

  - **docs/experiments.md** records every result under them. A rename does not break
    anything loudly -- it starts a *new* series and leaves the old one ending at
    whatever epoch the rename landed on.
  - **``--monitor``** names one on the command line, and it decides which checkpoint
    survives (see the ``checkpoint-selection`` findings). A name that drifts turns
    checkpoint selection into a ``KeyError`` an hour into a run, at best.
  - Any chart or comparison script pointed at a stored run.

If one of these assertions fails, the question is not "what is the new name" but
"**what did I rename, and does an existing run's chart still resolve**". Extending the
set is ordinary; changing a name that is already there is a migration.

That this file can exist at all is the point of typing the results: ``SCALARS`` is a
``ClassVar``, so the full name set is knowable from the types with nothing scored.
Under the old ``dict[str, float]`` results the only way to learn these names was to
train an epoch and read MLflow.
"""

from stemify.training.validation import (
    CUES_RESULT_TYPES,
    NOTE_LOG_KEYS,
    harness_metric_names,
)

# Every metric the async validator logs for one condition, exactly as MLflow sees it.
EXPECTED = {
    "note/clean/f1",
    "note/clean/precision",
    "note/clean/recall",
    "note/clean/fragmentation",
    "note/clean/fp",
    "note/clean/fp_rest",
    "note/clean/fp_interior_offpitch",
    "cues/clean/onset_ap",
    "cues/clean/onset_best_threshold",
    "cues/clean/onset_best_f1",
    "cues/clean/onset_positive_rate",
    "cues/clean/voicing_auroc",
    "cues/clean/voicing_best_threshold",
    "cues/clean/voicing_best_f1",
}


def test_the_logged_names_are_what_they_have_always_been():
    assert set(harness_metric_names(["clean"])) == EXPECTED


def test_every_condition_gets_the_same_record():
    """A run comparing clean against separated is comparing two series that must be
    the same shape; the gap between them is the verdict on separation."""
    names = harness_metric_names(["clean", "separated"])
    clean = {n for n in names if "/clean/" in n}
    separated = {n for n in names if "/separated/" in n}
    assert {n.replace("/clean/", "/X/") for n in clean} == {
        n.replace("/separated/", "/X/") for n in separated
    }


def test_no_name_is_produced_twice():
    names = harness_metric_names(["clean", "separated"])
    assert len(names) == len(set(names))


def test_the_note_projection_is_exhaustive_not_a_fallback():
    """``NOTE_LOG_KEYS`` is a closed table, so a metric key that disappears raises at
    the log site. It used to be ``_NOTE_KEY.get(key, key)``, which quietly logged a
    metric's internal name the moment the two drifted -- silently creating a series
    under a name no chart was pointed at."""
    from stemify.metrics import Fragmentation, PRF

    reported = set(PRF().compute([[]], [[]]).scalars) | set(
        Fragmentation().compute([[]], [[]]).scalars
    )
    assert set(NOTE_LOG_KEYS) <= reported, (
        f"NOTE_LOG_KEYS projects {set(NOTE_LOG_KEYS) - reported}, which no note "
        "metric reports any more."
    )


def test_the_cues_names_come_from_the_result_types():
    """Not a hand-written list beside them -- the types are the source, which is what
    keeps this file from being a second thing to update."""
    declared = {key for t in CUES_RESULT_TYPES for key in t.SCALARS}
    logged = {n.split("/")[-1] for n in harness_metric_names(["clean"]) if "cues/" in n}
    assert declared == logged
