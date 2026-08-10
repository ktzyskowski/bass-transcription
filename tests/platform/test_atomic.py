"""``atomic_write`` -- the guarantee that a reader sees a whole file or none.

Everything on disk in this project is a memoised expensive computation, and the failure
being prevented is silent: a half-written file is a *cache hit* on the next run, so the
cost of getting this wrong is not a crash but a run trained on a truncated CQT. The
tests below are the failure modes, not the happy path.
"""

import os

import pytest

from stemify.platform.atomic import atomic_write


def test_the_payload_lands_at_the_destination(tmp_path):
    dest = tmp_path / "out.txt"
    atomic_write(dest, lambda partial: partial.write_text("payload"))
    assert dest.read_text() == "payload"


def test_the_parent_directory_is_created(tmp_path):
    """Callers hand over a path inside a cache layout that may not exist yet -- the
    features/contours/transforms directories are named by ``Cache``, not pre-made."""
    dest = tmp_path / "a" / "b" / "c" / "out.txt"
    atomic_write(dest, lambda partial: partial.write_text("payload"))
    assert dest.read_text() == "payload"


def test_a_failed_write_leaves_no_destination_at_all(tmp_path):
    """The central guarantee. The writer got as far as putting bytes down before
    raising, and none of them may be visible under the destination name."""
    dest = tmp_path / "out.txt"

    def write_then_fail(partial):
        partial.write_text("half a fi")
        raise RuntimeError("interrupted mid-track")

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write(dest, write_then_fail)
    assert not dest.exists()


def test_a_failed_write_leaves_no_partial_behind_either(tmp_path):
    """A stray ``.partial`` is not a correctness bug but it is an unbounded one: these
    caches hold separated audio, and a run that dies once per launch accumulates
    hundreds of megabytes nothing will ever collect."""
    dest = tmp_path / "out.txt"

    def boom(partial):
        partial.write_text("half")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write(dest, boom)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_destination_survives_a_failed_rewrite(tmp_path):
    """Recomputing over a warm cache must not be able to destroy what is already there.
    ``replace`` is never reached, so the old bytes stay -- a stale entry beats no entry,
    and the caller can retry."""
    dest = tmp_path / "out.txt"
    dest.write_text("the good copy")

    with pytest.raises(RuntimeError):
        atomic_write(dest, lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert dest.read_text() == "the good copy"


def test_an_existing_destination_is_replaced_on_success(tmp_path):
    """``Path.replace`` rather than ``rename``: on some platforms the latter refuses an
    existing target, which would turn every cache refresh into an error."""
    dest = tmp_path / "out.txt"
    dest.write_text("stale")
    atomic_write(dest, lambda partial: partial.write_text("fresh"))
    assert dest.read_text() == "fresh"


def test_the_writer_is_handed_a_path_that_is_not_the_destination(tmp_path):
    """What makes the write atomic: the payload is built somewhere else entirely, so
    there is no window in which ``dest`` names an incomplete file. The pid in the temp
    name is what keeps two concurrent producers -- a pre-warm pass beside a run -- from
    writing into each other's buffer."""
    dest = tmp_path / "out.txt"
    seen = []

    def record(partial):
        seen.append(partial)
        partial.write_text("payload")

    atomic_write(dest, record)
    (partial,) = seen
    assert partial != dest
    assert partial.parent == dest.parent  # same filesystem, or the rename is not atomic
    assert str(os.getpid()) in partial.name
